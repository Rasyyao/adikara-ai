import http.client
import json
import os
import re
import urllib.error
import urllib.request

import embed

TIMEOUT = 300
NUM_CTX = 8192
MAX_TOKENS = 6144
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_QWEN_MODEL = "qwen/qwen3.8-flash"
DEFAULT_CHAT_MODEL = DEFAULT_GEMINI_MODEL

COMPARE_MODELS: tuple[str, ...] = (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_GEMINI_MODEL,
)

MODEL_ALIASES: dict[str, str] = {
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "deepseek-v4.1-flash": DEFAULT_DEEPSEEK_MODEL,
    "deepseek v4.1 flash": DEFAULT_DEEPSEEK_MODEL,
    "deepseek v 4.1 flash": DEFAULT_DEEPSEEK_MODEL,
    "deepseek-v-4.1-flash": DEFAULT_DEEPSEEK_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "qwen-3.8-flash": DEFAULT_QWEN_MODEL,
    "qwen 3.8 flash": DEFAULT_QWEN_MODEL,
    "qwen3.8-flash": DEFAULT_QWEN_MODEL,
    "qwen3.8 flash": DEFAULT_QWEN_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "gemini-3.5-flash-lite": DEFAULT_GEMINI_MODEL,
    "gemini 3.5-flash-lite": DEFAULT_GEMINI_MODEL,
    "gemini 3.5 flash lite": DEFAULT_GEMINI_MODEL,
}

OPENROUTER_PREFIXES = (
    "openrouter/",
    "deepseek/",
    "qwen/",
    "google/",
    "anthropic/",
    "meta-llama/",
    "mistralai/",
    "openai/",
    "x-ai/",
)


class LLMError(Exception):
    """Pesan aman untuk ditampilkan ke pengguna."""


def resolve_model(model: str) -> str:
    cleaned = model.strip()
    key = re.sub(r"\s+", " ", cleaned.lower())
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    key_hyphen = key.replace(" ", "-")
    if key_hyphen in MODEL_ALIASES:
        return MODEL_ALIASES[key_hyphen]
    return cleaned


def gemini_api_key() -> str | None:
    embed.load_env()
    return (
        os.environ.get("GEMINI_API", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or None
    )


def openrouter_api_key() -> str | None:
    embed.load_env()
    return (
        os.environ.get("OPEN_ROUTER", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
        or None
    )


def _is_openrouter(model: str, endpoint: str) -> bool:
    ep = endpoint.strip().lower()
    if ep in ("openrouter", "open_router"):
        return True
    m = resolve_model(model).lower()
    if m.startswith(OPENROUTER_PREFIXES):
        return True
    return False


def _is_gemini(model: str, endpoint: str) -> bool:
    if _is_openrouter(model, endpoint):
        return False
    ep = endpoint.strip().lower()
    m = resolve_model(model).lower()
    return ep == "gemini" or m.startswith(("gemini", "models/gemini"))


def provider_name(model: str, endpoint: str = "local") -> str:
    if _is_openrouter(model, endpoint):
        return "openrouter"
    if _is_gemini(model, endpoint):
        return "gemini"
    return endpoint or "local"


def _gemini_payload(messages: list[dict], json_mode: bool = False) -> dict:
    system_parts = []
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_parts.append({"text": text})
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})
    payload: dict = {"contents": contents}
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
    return payload


def _chat_gemini(model: str, messages: list[dict], json_mode: bool = False) -> str:
    api_key = gemini_api_key()
    if not api_key:
        raise LLMError("GEMINI_API belum diatur di .env.")
    clean_model = model.removeprefix("models/").strip() or DEFAULT_GEMINI_MODEL
    url = f"{GEMINI_BASE_URL}/models/{clean_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    body = json.dumps(_gemini_payload(messages, json_mode=json_mode)).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()
        raise LLMError(f"Gemini menolak permintaan untuk model '{clean_model}' (HTTP {e.code}): {detail}") from e
    except OSError as e:
        raise LLMError(f"Tidak bisa terhubung ke Gemini API: {getattr(e, 'reason', e)}") from e
    except (http.client.HTTPException, ValueError) as e:
        raise LLMError(f"Respons dari Gemini tidak bisa dibaca: {e}") from e

    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise LLMError(f"Gemini tidak mengembalikan kandidat jawaban untuk model '{clean_model}'.")
    content_obj = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content_obj.get("parts") if isinstance(content_obj, dict) else None
    if not isinstance(parts, list):
        raise LLMError(f"Format jawaban Gemini untuk model '{clean_model}' tidak dikenali.")

    text = "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str)
    )
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise LLMError("Model Gemini mengembalikan jawaban kosong.")
    return text


def _chat_ollama(model: str, messages: list[dict], endpoint: str = "local", json_mode: bool = False) -> str:
    try:
        base, api_key, key_var = embed._endpoint_config(endpoint)
        headers = embed._auth_headers(base, api_key, key_var)
    except embed.EmbedError as e:
        raise LLMError(str(e)) from e
    req_body: dict = {"model": model, "messages": messages, "stream": False,
                      "options": {"num_ctx": NUM_CTX}}
    if json_mode:
        req_body["format"] = "json"
    body = json.dumps(req_body).encode("utf-8")
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


def _chat_openrouter(model: str, messages: list[dict], json_mode: bool = False) -> str:
    api_key = openrouter_api_key()
    if not api_key:
        raise LLMError("OPEN_ROUTER belum diatur di .env.")
    clean_model = resolve_model(model).removeprefix("openrouter/").strip()
    if clean_model.startswith("gemini"):
        clean_model = f"google/{clean_model}"
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req_body: dict = {"model": clean_model, "messages": messages, "max_tokens": MAX_TOKENS}
    if json_mode:
        req_body["response_format"] = {"type": "json_object"}
    body = json.dumps(req_body).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()
        raise LLMError(f"OpenRouter menolak permintaan untuk model '{clean_model}' (HTTP {e.code}): {detail}") from e
    except OSError as e:
        raise LLMError(f"Tidak bisa terhubung ke OpenRouter API: {getattr(e, 'reason', e)}") from e
    except (http.client.HTTPException, ValueError) as e:
        raise LLMError(f"Respons dari OpenRouter tidak bisa dibaca: {e}") from e

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise LLMError(f"OpenRouter tidak mengembalikan pilihan jawaban untuk model '{clean_model}'.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str):
        raise LLMError(f"Format jawaban OpenRouter untuk model '{clean_model}' tidak dikenali.")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        raise LLMError(f"Model OpenRouter '{clean_model}' mengembalikan jawaban kosong.")
    return content


def chat(model: str, messages: list[dict], endpoint: str = "local", json_mode: bool = False) -> str:
    resolved = resolve_model(model)
    if _is_openrouter(resolved, endpoint):
        return _chat_openrouter(resolved, messages, json_mode=json_mode)
    if _is_gemini(resolved, endpoint):
        return _chat_gemini(resolved, messages, json_mode=json_mode)
    return _chat_ollama(resolved, messages, endpoint, json_mode=json_mode)
