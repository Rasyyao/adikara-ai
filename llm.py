"""Jawaban chat dari model Ollama (/api/chat). Endpoint dan kunci API memakai konfigurasi yang sama dengan embed.py."""
import http.client
import json
import re
import urllib.error
import urllib.request

import embed

TIMEOUT = 300
NUM_CTX = 8192


class LLMError(Exception):
    """Pesan aman untuk ditampilkan ke pengguna."""


def chat(model: str, messages: list[dict], endpoint: str = "local") -> str:
    try:
        base, api_key, key_var = embed._endpoint_config(endpoint)
        headers = embed._auth_headers(base, api_key, key_var)
    except embed.EmbedError as e:
        raise LLMError(str(e)) from e
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"num_ctx": NUM_CTX}}).encode("utf-8")
    request = urllib.request.Request(f"{base}/api/chat", data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()
        raise LLMError(f"Ollama menolak permintaan untuk model '{model}' (HTTP {e.code}): {detail}") from e
    except OSError as e:
        raise LLMError(f"Tidak bisa terhubung ke Ollama di {base}: {getattr(e, 'reason', e)}") from e
    except (http.client.HTTPException, ValueError) as e:
        raise LLMError(f"Respons dari Ollama tidak bisa dibaca: {e}") from e

    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError(f"Ollama tidak mengembalikan jawaban untuk model '{model}'.")
    # model "thinking" menaruh penalarannya di <think>...</think>; guru hanya perlu jawabannya
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        raise LLMError("Model mengembalikan jawaban kosong.")
    return content
