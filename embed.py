import hashlib
import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_OLLAMA_URL = "http://localhost:11434"
FINAL_MODEL = "qwen3-embedding:0.6b"   
BATCH_SIZE = 32
TIMEOUT = 300       
NUM_CTX = 4096
CACHE_DIR = Path("cache")


class EmbedError(Exception):
    """Pesan aman untuk ditampilkan ke pengguna."""


# Prefixes are model-specific, and a wrong or missing one silently lowers retrieval quality.
# Every new model must be checked against its model card before being added here.
PREFIXES = {
    "nomic-embed-text": {"document": "search_document: ", "query": "search_query: "},
    "qwen3-embedding": {
        "document": "",
        "query": "Instruct: Given a question, retrieve passages that answer it\nQuery: ",
    },
}
NO_PREFIX = {"document": "", "query": ""}


def _prefix(model: str, kind: str) -> str:
    # match on the name before ":" so tags like "qwen3-embedding:0.6b" work
    return PREFIXES.get(model.split(":")[0], NO_PREFIX)[kind]


def _key(sent_text: str) -> str:
    return hashlib.sha256(sent_text.encode("utf-8")).hexdigest()


def load_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ; variables already set always win."""
    path = Path(path) if path else Path(__file__).parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _endpoint_config(endpoint: str) -> tuple[str, str | None, str]:
    """Return (base url, api key or None, name of the key variable) for a named endpoint."""
    if endpoint == "local":
        url_var, key_var = "OLLAMA_URL", "OLLAMA_API_KEY"
    else:
        name = re.sub(r"\W", "_", endpoint).upper()
        url_var, key_var = f"OLLAMA_{name}_URL", f"OLLAMA_{name}_API_KEY"
    url = os.environ.get(url_var, "").strip()
    if not url:
        if endpoint != "local":
            raise EmbedError(f"Endpoint '{endpoint}' belum dikonfigurasi: isi {url_var} di .env.")
        url = DEFAULT_OLLAMA_URL
    return url.rstrip("/"), os.environ.get(key_var, "").strip() or None, key_var


def _cache_path(model: str, endpoint: str = "local") -> Path:
    name = model.replace(":", "_").replace("/", "_")
    if endpoint != "local":          # local keeps the plain name so existing caches stay valid
        name += "@" + endpoint.replace(":", "_").replace("/", "_")
    return CACHE_DIR / (name + ".json")


def _load_cache(model: str, endpoint: str = "local") -> dict[str, list[float]]:
    path = _cache_path(model, endpoint)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"Peringatan: cache {path} rusak, akan dibuat ulang.")
        return {}


def _save_cache(model: str, cache: dict[str, list[float]], endpoint: str = "local") -> None:
    path = _cache_path(model, endpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, path)            # atomic: a crash never leaves a half-written cache


def _normalize(vectors: list) -> list[list[float]]:
    try:
        arr = np.asarray(vectors, dtype=np.float64)
    except ValueError as e:
        raise EmbedError("Ollama mengembalikan vektor dengan panjang yang tidak seragam.") from e
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise EmbedError("Format vektor dari Ollama tidak dikenali.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbedError("Ollama mengembalikan vektor nol; teks mungkin kosong.")
    return (arr / norms).tolist()


def _post_embed(model: str, texts: list[str], endpoint: str = "local") -> list[list[float]]:
    base, api_key, key_var = _endpoint_config(endpoint)
    headers = {"Content-Type": "application/json"}
    if api_key:
        host = urllib.parse.urlparse(base)
        if host.scheme != "https" and host.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise EmbedError(
                f"Kunci API ({key_var}) tidak dikirim lewat koneksi tanpa enkripsi ke {base}. "
                f"Gunakan alamat https://.")
        headers["Authorization"] = f"Bearer {api_key}"     # never print or log this header
    body = json.dumps({
        "model": model,
        "input": texts,
        "truncate": False,           # fail loudly instead of silently cutting long chunks
        "options": {"num_ctx": NUM_CTX},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/api/embed", data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:      # must come before OSError: HTTPError is a subclass
        detail = e.read().decode("utf-8", errors="replace").strip()
        hint = (f"Periksa {key_var} di .env." if e.code in (401, 403)
                else f"Jika modelnya belum diunduh, jalankan `ollama pull {model}`.")
        raise EmbedError(
            f"Ollama menolak permintaan untuk model '{model}' (HTTP {e.code}): {detail}\n{hint}") from e
    except OSError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, TimeoutError):
            raise EmbedError(f"Ollama tidak merespons dalam {TIMEOUT} detik ({base}).") from e
        how = ("Pastikan Ollama berjalan dengan `ollama serve`." if endpoint == "local"
               else "Periksa URL endpoint dan koneksi jaringan.")
        raise EmbedError(f"Tidak bisa terhubung ke Ollama di {base}. {how} ({reason})") from e
    except (http.client.HTTPException, ValueError) as e:
        raise EmbedError(f"Respons dari Ollama tidak bisa dibaca: {e}") from e

    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbedError(
            f"Ollama mengembalikan jumlah vektor yang tidak sesuai untuk model '{model}'.")
    return _normalize(embeddings)


def _embed(model: str, texts: list[str], kind: str, verbose: bool,
           endpoint: str = "local") -> list[list[float]]:
    if not texts:
        return []
    prefix = _prefix(model, kind)
    sent = [prefix + t for t in texts]          # the prefixed text is what gets hashed and sent
    keys = [_key(s) for s in sent]

    cache = _load_cache(model, endpoint)
    missing = {k: s for k, s in zip(keys, sent) if k not in cache}   # also dedupes repeats
    items = list(missing.items())
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        vectors = _post_embed(model, [s for _, s in batch], endpoint)
        cache.update(zip((k for k, _ in batch), vectors))
        _save_cache(model, cache, endpoint)
        if verbose:
            print(f"  {model}: {start + len(batch)}/{len(items)} teks selesai di-embed")
    return [cache[k] for k in keys]


def count_uncached(model: str, texts: list[str], kind: str = "document",
                   endpoint: str = "local") -> int:
    """Jumlah teks yang belum ada di cache (dipakai untuk melaporkan sumber waktu embed)."""
    cache = _load_cache(model, endpoint)
    prefix = _prefix(model, kind)
    return sum(1 for t in texts if _key(prefix + t) not in cache)


def embed_documents(model: str, texts: list[str], endpoint: str = "local") -> list[list[float]]:
    return _embed(model, texts, "document", verbose=True, endpoint=endpoint)


def embed_query(model: str, query: str, endpoint: str = "local") -> list[float]:
    return _embed(model, [query], "query", verbose=False, endpoint=endpoint)[0]
