"""Pekerja antrean: mendengarkan NOTIFY 'jobs', mengambil job EXTRACT, mengindeks materi.

Alur: API menyisipkan baris jobs (queued) lalu `NOTIFY jobs`. Pekerja bangun, mengklaim satu job dengan
FOR UPDATE SKIP LOCKED (aman dijalankan lebih dari satu pekerja), dan memprosesnya. NOTIFY hilang bila pekerja
mati, jadi setiap bangun (dan tiap SWEEP_SECONDS) pekerja menguras semua job yang menunggu.
"""
import json
import logging
import os
import tempfile

import embed
import llm
import prompt
import storage
import store
from extract import ExtractionError

log = logging.getLogger("worker")

CHANNEL = "jobs"
SWEEP_SECONDS = 30
STALE_AFTER = "10 minutes"   # job 'running' yang lebih lama dari ini dianggap ditinggal pekerja yang mati
MAX_ATTEMPTS = 3

RETRY_AFTER = "15 seconds"   # jeda sebelum job yang gagal sementara boleh diambil lagi

# extract: kapan saja. chat_reply: tunggu sampai semua berkas chat itu selesai dibaca (agar jawabannya memakai berkas
# yang baru diunggah) dan sampai balasan sebelumnya di chat yang sama selesai (agar urutan jawaban benar).
CLAIM = f"""
UPDATE jobs SET status = 'running', locked_at = now(), attempts = attempts + 1
WHERE id = (
    SELECT j.id FROM jobs j
    WHERE j.type IN ('extract', 'chat_reply')
      AND (j.status = 'queued' OR (j.status = 'running' AND j.locked_at < now() - interval '{STALE_AFTER}'))
      AND (j.locked_at IS NULL OR j.locked_at < now() - interval '{RETRY_AFTER}')
      AND (j.type = 'extract' OR (
            NOT EXISTS (SELECT 1 FROM chat_materials cm JOIN materials m ON m.id = cm.material_id
                        WHERE cm.chat_id = (j.payload->>'chatId')::uuid AND m.status = 'uploaded')
        AND NOT EXISTS (SELECT 1 FROM jobs r WHERE r.id <> j.id AND r.type = 'chat_reply' AND r.status = 'running'
                        AND r.payload->>'chatId' = j.payload->>'chatId'
                        AND r.locked_at > now() - interval '{STALE_AFTER}')))
    ORDER BY j.created_at
    FOR UPDATE OF j SKIP LOCKED
    LIMIT 1)
RETURNING id::text, payload, attempts, type::text
"""

# Berasal dari berkas itu sendiri; mengulang tidak akan membantu dan pesannya aman untuk guru.
FINAL_ERRORS = (ExtractionError, storage.StorageError, store.StoreError)
# Berasal dari layanan AI (Ollama): pesannya memuat alamat server, jadi guru hanya melihat pesan umum.
SERVICE_ERRORS = (embed.EmbedError, llm.LLMError)
SERVICE_DOWN = "Layanan AI sedang bermasalah. Coba lagi beberapa saat lagi."


def claim(conn) -> dict | None:
    row = conn.execute(CLAIM).fetchone()
    if row is None:
        return None
    payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    return {"id": row[0], "payload": payload, "attempts": row[2], "type": row[3]}


def _finish(conn, job_id: str, status: str, error: str | None = None, result_ref: str | None = None) -> None:
    # 'queued' keeps locked_at: the claim query then waits RETRY_AFTER before trying the job again
    conn.execute(
        """UPDATE jobs SET status = %s, error = %s, result_ref = %s,
                  locked_at = CASE WHEN %s = 'queued' THEN now() END,
                  finished_at = CASE WHEN %s IN ('done', 'failed') THEN now() END
           WHERE id = %s""",
        (status, error, result_ref, status, status, job_id))


def _process_extract(conn, material_id: str, model: str, endpoint: str) -> None:
    row = conn.execute("SELECT file_path FROM materials WHERE id = %s", (material_id,)).fetchone()
    if row is None:
        raise store.StoreError("Materi sudah dihapus.")
    with tempfile.TemporaryDirectory(prefix="adikara-") as tmp:
        path = storage.download(row[0], tmp)
        store.index_material(conn, material_id, path, model, endpoint)


HISTORY_MESSAGES = 12     # pesan terakhir yang dikirim ke model
CONTEXT_CHUNKS = 5
NO_MODEL = "Model AI untuk chat belum diatur di server (CHAT_MODEL). Hubungi admin."
REPLY_FAILED = "Maaf, balasan gagal dibuat. Coba kirim ulang pesan Anda."


def _chat_config() -> tuple[str | None, str]:
    raw_model = os.environ.get("CHAT_MODEL", "").strip()
    model = llm.resolve_model(raw_model) if raw_model else None
    endpoint = os.environ.get("CHAT_ENDPOINT", "local").strip() or "local"
    if model is None:
        if os.environ.get("GEMINI_API", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip():
            return llm.DEFAULT_GEMINI_MODEL, "gemini"
        if (
            os.environ.get("OPEN_ROUTER", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
        ):
            return llm.DEFAULT_DEEPSEEK_MODEL, "openrouter"
    return model, endpoint


def _save_reply(conn, chat_id: str, content: str, sources: list[str] | None = None) -> None:
    conn.execute(
        """INSERT INTO chat_messages (id, chat_id, role, content, sources)
           VALUES (gen_random_uuid(), %s, 'assistant', %s, %s)""",
        (chat_id, content, json.dumps(sources) if sources else None))


def _process_chat_reply(conn, payload: dict, embed_model: str, embed_endpoint: str) -> None:
    chat_id, message_id = payload.get("chatId"), payload.get("messageId")
    row = conn.execute("SELECT created_at FROM chat_messages WHERE id = %s AND chat_id = %s",
                       (message_id, chat_id)).fetchone()
    if row is None:
        raise store.StoreError("Pesan sudah dihapus.")
    chat_model, chat_endpoint = _chat_config()
    if chat_model is None:
        _save_reply(conn, chat_id, NO_MODEL)
        return

    history = [{"role": r[0], "content": r[1]} for r in reversed(conn.execute(
        """SELECT role::text, content FROM chat_messages
           WHERE chat_id = %s AND (created_at, id) <= (%s, %s)
           ORDER BY created_at DESC, id DESC LIMIT %s""",
        (chat_id, row[0], message_id, HISTORY_MESSAGES)).fetchall())]
    failed = [{"title": r[0], "error": r[1]} for r in conn.execute(
        """SELECT m.title, m.error FROM chat_materials cm JOIN materials m ON m.id = cm.material_id
           WHERE cm.chat_id = %s AND m.status = 'failed'""", (chat_id,)).fetchall()]
    ready = conn.execute(
        """SELECT 1 FROM chat_materials cm JOIN materials m ON m.id = cm.material_id
           WHERE cm.chat_id = %s AND m.status = 'extracted' LIMIT 1""", (chat_id,)).fetchone()
    chunks = (store.retrieve(conn, chat_id, history[-1]["content"], CONTEXT_CHUNKS, embed_model, embed_endpoint)
              if ready else [])
    web_results = prompt.search_web_context(history[-1]["content"]) if chat_model != "fake-chat" else []

    answer = llm.chat(chat_model, prompt.build_messages(history, chunks, failed, web_results=web_results), chat_endpoint)
    _save_reply(conn, chat_id, answer, [c["id"] for c in chunks])


def process_one(conn, model: str = embed.FINAL_MODEL, endpoint: str = "local") -> bool:
    """Klaim dan proses satu job. False bila tidak ada job. Koneksi harus autocommit agar klaim langsung terlihat."""
    job = claim(conn)
    if job is None:
        return False
    payload = job["payload"]
    material_id = payload.get("materialId") if job["type"] == "extract" else None
    log.info("job %s (%s) percobaan %d", job["id"], job["type"], job["attempts"])
    try:
        if job["type"] == "chat_reply":
            _process_chat_reply(conn, payload, model, endpoint)
        else:
            if not material_id:
                raise store.StoreError("Job tidak memuat materialId.")
            _process_extract(conn, material_id, model, endpoint)
    except FINAL_ERRORS as e:
        _fail(conn, job, material_id, str(e))
    except SERVICE_ERRORS as e:
        log.error("job %s: layanan AI gagal: %s", job["id"], e)
        _fail(conn, job, material_id, SERVICE_DOWN, detail=str(e))
    except Exception:
        log.exception("job %s gagal tak terduga", job["id"])
        if job["attempts"] >= MAX_ATTEMPTS:
            _fail(conn, job, material_id, "Berkas tidak bisa diproses. Coba unggah ulang."
                  if material_id else REPLY_FAILED)
        else:
            _finish(conn, job["id"], "queued", error="percobaan gagal, akan diulang")
    else:
        if material_id:
            conn.execute("UPDATE materials SET status = 'extracted', error = NULL WHERE id = %s", (material_id,))
        _finish(conn, job["id"], "done", result_ref=material_id)
    return True


def _fail(conn, job: dict, material_id: str | None, message: str, detail: str | None = None) -> None:
    """message dilihat guru; detail (teknis) hanya disimpan di jobs.error."""
    if material_id:
        # the teacher sees why: the API reads this column
        conn.execute("UPDATE materials SET status = 'failed', error = %s WHERE id = %s", (message, material_id))
    elif job["type"] == "chat_reply" and job["payload"].get("chatId"):
        # nobody is waiting on the job itself, so the teacher must find the failure in the chat
        _save_reply(conn, job["payload"]["chatId"], REPLY_FAILED)
    _finish(conn, job["id"], "failed", error=detail or message)


def drain(conn, model: str = embed.FINAL_MODEL, endpoint: str = "local") -> int:
    done = 0
    while process_one(conn, model, endpoint):
        done += 1
    return done


def run(model: str = embed.FINAL_MODEL, endpoint: str = "local") -> None:
    listener = store.connect(autocommit=True)
    worker = store.connect(autocommit=True)
    listener.execute(f"LISTEN {CHANNEL}")
    log.info("menunggu job (model %s)", model)
    drain(worker, model, endpoint)                       # job yang menunggu saat pekerja mati
    while True:
        for _ in listener.notifies(timeout=SWEEP_SECONDS, stop_after=1):
            pass                                          # bangun karena NOTIFY
        drain(worker, model, endpoint)                    # juga jalan saat timeout: sweep


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
