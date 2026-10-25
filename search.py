"""Uji alur RAG penuh tanpa autentikasi API / R2:
(berkas lokal -> extract -> chunk -> embed -> simpan ke Postgres)
-> embed pertanyaan -> cari top-k di pgvector -> susun prompt -> jawab dengan LLM (Gemini / Ollama).

Pemakaian:
  python search.py
  python search.py "Apa itu perulangan?"
  python search.py "Apa itu perulangan?" --compare
  python search.py "Apa itu perulangan?" --chat-models "deepseek v4.1 flash" "qwen 3.8 flash" "gemini 3.5-flash-lite"
  python search.py "Apa itu array 2 dimensi?" --file data/test-modul.pdf --k 5
  python search.py "Apa itu perulangan?" --no-llm
  python search.py "Apa itu perulangan?" --material <material-uuid> --k 5
  python search.py "Apa itu perulangan?" --chat <chat-uuid> --k 5 --min-score 0.3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import embed
import llm
import prompt
import store
import worker
from enqueue import ensure_teacher
from extract import ExtractionError

DEFAULT_FILE = Path(__file__).parent / "data" / "test-modul.pdf"
DEFAULT_QUERY = "Apa itu perulangan (looping) dalam pemrograman?"


def index_local_file(conn, path: str | Path, model: str = embed.FINAL_MODEL,
                     endpoint: str = "local") -> tuple[str, str]:
    """Pastikan berkas lokal terindeks di Postgres tanpa butuh login/R2. Mengembalikan (material_id, title)."""
    path = Path(path)
    if not path.is_file():
        raise store.StoreError(f"Berkas tidak ditemukan: {path}")
    teacher_id = ensure_teacher(conn)
    title = path.stem
    row = conn.execute(
        "SELECT id::text FROM materials WHERE teacher_id = %s AND title = %s ORDER BY created_at DESC LIMIT 1",
        (teacher_id, title),
    ).fetchone()
    if row is None:
        material_id = conn.execute(
            """INSERT INTO materials (id, teacher_id, title, file_path, file_type, status)
               VALUES (gen_random_uuid(), %s, %s, %s, %s, 'uploaded') RETURNING id::text""",
            (teacher_id, title, str(path), path.suffix.lower().lstrip(".")),
        ).fetchone()[0]
    else:
        material_id = row[0]

    has_chunks = conn.execute(
        "SELECT 1 FROM material_chunks WHERE material_id = %s AND embedding_model = %s LIMIT 1",
        (material_id, model),
    ).fetchone()
    if not has_chunks:
        n = store.index_material(conn, material_id, path, model=model, endpoint=endpoint)
        conn.execute("UPDATE materials SET status = 'extracted', error = NULL WHERE id = %s", (material_id,))
        print(f"[Index] '{path.name}' berhasil diindeks ke Postgres ({n} chunk, model {model}).\n")
    return material_id, title


def _latest_material(conn, model: str) -> tuple[str, str] | None:
    row = conn.execute(
        """SELECT m.id::text, m.title
           FROM materials m
           WHERE EXISTS (
               SELECT 1 FROM material_chunks mc
               WHERE mc.material_id = m.id AND mc.embedding_model = %s
           )
           ORDER BY m.created_at DESC
           LIMIT 1""",
        (model,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def format_results(query: str, hits: list[dict], target_label: str,
                   web_results: list[dict] | None = None) -> str:
    lines = [
        f"Pertanyaan : {query}",
        f"Target RAG : {target_label}",
        f"Hasil RAG  : {len(hits)} chunk ditemukan (acuan topik slide)\n",
    ]
    if not hits:
        lines.append("(Tidak ada chunk yang cocok atau melewati batas min_score)\n")
    else:
        for rank, h in enumerate(hits, start=1):
            title = f"{h['title']} — " if h.get("title") else ""
            lines.append(
                f"--- #{rank} | skor: {h['score']:.4f} | {title}{h['label']} | id: {h['id']} ---\n"
                f"{h['text']}\n"
            )
    if web_results is not None:
        lines.append(f"===== REFERENSI INTERNET ({len(web_results)} sumber ditemukan) =====")
        for idx, w in enumerate(web_results, start=1):
            lines.append(f"[{idx}] {w['title']} — {w.get('url', '-')}\n    {w['snippet']}\n")
    return "\n".join(lines)


def run_search(conn, query: str, material_id: str | None = None, chat_id: str | None = None,
               file_path: str | Path | None = None, k: int = 5, min_score: float = 0.0,
               model: str = embed.FINAL_MODEL, endpoint: str = "local") -> tuple[list[dict], str]:
    if sum(bool(x) for x in (chat_id, material_id, file_path)) > 1:
        raise store.StoreError("Pilih salah satu: --file, --material, atau --chat.")
    if chat_id:
        hits = store.retrieve(conn, chat_id, query, k=k, model=model, endpoint=endpoint, min_score=min_score)
        return hits, f"chat {chat_id}"
    if file_path:
        material_id, title = index_local_file(conn, file_path, model=model, endpoint=endpoint)
        target = f"material {material_id} ({title})"
    elif not material_id:
        latest = _latest_material(conn, model)
        if latest is not None:
            material_id, title = latest
            target = f"material {material_id} ({title})"
        elif DEFAULT_FILE.is_file():
            material_id, title = index_local_file(conn, DEFAULT_FILE, model=model, endpoint=endpoint)
            target = f"material {material_id} ({title})"
        else:
            raise store.StoreError(
                f"Belum ada materi yang memiliki chunk dengan model '{model}'. "
                "Gunakan --file <berkas> untuk mengindeks berkas lokal."
            )
    else:
        target = f"material {material_id}"
    hits = store.retrieve_material(
        conn, material_id, query, k=k, model=model, endpoint=endpoint, min_score=min_score
    )
    return hits, target


def answer_with_llm(query: str, hits: list[dict], chat_model: str | None = None,
                    chat_endpoint: str | None = None,
                    web_results: list[dict] | None = None) -> tuple[str, str]:
    """Susun prompt dari chunk top-k + referensi internet, lalu panggil LLM. Mengembalikan (model, jawaban)."""
    default_model, default_endpoint = worker._chat_config()
    raw_model = chat_model or default_model or llm.DEFAULT_CHAT_MODEL
    model = llm.resolve_model(raw_model)
    endpoint = chat_endpoint or (default_endpoint if not chat_model else "local")
    if web_results is None:
        web_results = prompt.search_web_context(query)
    messages = prompt.build_messages(
        history=[{"role": "user", "content": query}],
        chunks=hits,
        failed=[],
        web_results=web_results,
    )
    return model, llm.chat(model, messages, endpoint=endpoint)


def compare_with_llm(query: str, hits: list[dict], models: list[str] | tuple[str, ...] | None = None,
                     mindmap_mode: bool = False,
                     web_results: list[dict] | None = None) -> list[dict]:
    """Jalankan pertanyaan yang sama ke beberapa model LLM dan ukur waktu respons serta hasilnya."""
    import mindmap

    if web_results is None and not mindmap_mode:
        web_results = prompt.search_web_context(query)
    target_models = list(dict.fromkeys(llm.resolve_model(m) for m in (models or llm.COMPARE_MODELS)))
    results: list[dict] = []
    for model in target_models:
        provider = llm.provider_name(model)
        start = time.perf_counter()
        try:
            if mindmap_mode:
                mm = mindmap.generate_mindmap(query, hits, model=model)
                elapsed = round(time.perf_counter() - start, 2)
                answer_text = json.dumps(mm, ensure_ascii=False, indent=2)
                results.append({
                    "model": model,
                    "provider": provider,
                    "status": "ok",
                    "seconds": elapsed,
                    "words": len(mm.get("nodes", [])),
                    "chars": len(answer_text),
                    "answer": answer_text,
                    "mindmap": mm,
                    "error": None,
                })
            else:
                used_model, answer = answer_with_llm(query, hits, chat_model=model, web_results=web_results)
                elapsed = round(time.perf_counter() - start, 2)
                results.append({
                    "model": used_model,
                    "provider": provider,
                    "status": "ok",
                    "seconds": elapsed,
                    "words": len(answer.split()),
                    "chars": len(answer),
                    "answer": answer,
                    "error": None,
                })
        except llm.LLMError as e:
            elapsed = round(time.perf_counter() - start, 2)
            results.append({
                "model": model,
                "provider": provider,
                "status": "error",
                "seconds": elapsed,
                "words": 0,
                "chars": 0,
                "answer": None,
                "error": str(e),
            })
    return results


def format_comparison(results: list[dict], mindmap_mode: bool = False) -> str:
    size_header = "Jumlah Node" if mindmap_mode else "Panjang Jawaban"
    headers = ["Model", "Provider", "Status", "Waktu (dtk)", size_header]
    rows = []
    for r in results:
        if r["status"] == "ok":
            size_val = f"{r['words']} node ({r['chars']} kar)" if mindmap_mode else f"{r['words']} kata ({r['chars']} kar)"
        else:
            size_val = "-"
        rows.append([
            r["model"],
            r["provider"],
            "OK" if r["status"] == "ok" else "GAGAL",
            f"{r['seconds']:.2f}",
            size_val,
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line_fmt = "  ".join("{:<%d}" % w for w in widths)
    out = [
        "===== PERBANDINGAN MODEL LLM =====",
        line_fmt.format(*headers),
        line_fmt.format(*("-" * w for w in widths)),
        *(line_fmt.format(*row) for row in rows),
        "",
    ]
    total = len(results)
    for idx, r in enumerate(results, start=1):
        header = f"===== [{idx}/{total}] {r['model']} ({r['provider']} | {r['seconds']:.2f} dtk) ====="
        body = r["answer"] if r["status"] == "ok" else f"[GAGAL] {r['error']}"
        out.append(f"{header}\n{body}\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Uji alur embed dokumen + embed pertanyaan + cari top-k di pgvector + jawab dengan LLM."
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY,
                        help=f"teks pertanyaan (bawaan: '{DEFAULT_QUERY}')")
    parser.add_argument("--file", help="berkas lokal (PDF/PPTX/DOCX) untuk diindeks langsung ke Postgres tanpa R2/auth")
    parser.add_argument("--material", help="UUID materi yang sudah ada di Postgres")
    parser.add_argument("--chat", help="UUID chat untuk mencari hanya di materi chat_materials")
    parser.add_argument("--k", type=int, default=5, help="jumlah top-k chunk (bawaan: 5)")
    parser.add_argument("--min-score", type=float, default=0.0, help="batas skor kemiripan minimum (bawaan: 0.0)")
    parser.add_argument("--model", default=embed.FINAL_MODEL, help=f"model embedding (bawaan: {embed.FINAL_MODEL})")
    parser.add_argument("--endpoint", default="local", help="endpoint Ollama untuk embedding (bawaan: local)")
    parser.add_argument("--chat-model", help=f"model LLM tunggal untuk jawaban (bawaan: {llm.DEFAULT_CHAT_MODEL})")
    parser.add_argument("--chat-models", nargs="+",
                        help="daftar model LLM untuk dibandingkan secara berdampingan")
    parser.add_argument("--compare", action="store_true",
                        help=f"bandingkan 3 model bawaan: {', '.join(llm.COMPARE_MODELS)}")
    parser.add_argument("--mindmap", action="store_true", help="hasilkan output JSON mindmap (tiap node berisi text)")
    parser.add_argument("--no-llm", action="store_true", help="hanya tampilkan hasil retrieval top-k tanpa memanggil LLM")
    parser.add_argument("--out", help="simpan hasil ke berkas JSON (opsional)")
    args = parser.parse_args(argv)

    if args.k < 1:
        sys.exit("Gagal: --k harus bilangan bulat positif.")

    try:
        with store.connect(autocommit=True) as conn:
            hits, target = run_search(
                conn,
                query=args.query,
                material_id=args.material,
                chat_id=args.chat,
                file_path=args.file,
                k=args.k,
                min_score=args.min_score,
                model=args.model,
                endpoint=args.endpoint,
            )
    except (store.StoreError, embed.EmbedError, ExtractionError) as e:
        sys.exit(f"Gagal: {e}")

    web_results = [] if (args.no_llm or args.mindmap) else prompt.search_web_context(args.query)
    print(format_results(args.query, hits, target, web_results=web_results if not args.no_llm and not args.mindmap else None))

    if not args.no_llm:
        if args.compare or args.chat_models:
            models_to_run = args.chat_models if args.chat_models else list(llm.COMPARE_MODELS)
            comp_results = compare_with_llm(
                args.query, hits, models=models_to_run, mindmap_mode=args.mindmap, web_results=web_results
            )
            print(format_comparison(comp_results, mindmap_mode=args.mindmap))
            if args.out:
                payload = {
                    "query": args.query,
                    "target": target,
                    "k": args.k,
                    "web_results": web_results,
                    "results": comp_results,
                }
                Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"[Tersimpan di {args.out}]", file=sys.stderr)
            if all(r["status"] != "ok" for r in comp_results):
                sys.exit("Gagal: semua model LLM gagal merespons.")
        else:
            try:
                if args.mindmap:
                    import mindmap
                    mm = mindmap.generate_mindmap(args.query, hits, model=args.chat_model)
                    formatted = json.dumps(mm, ensure_ascii=False, indent=2)
                    print(f"===== MINDMAP JSON ({mm['model']}) =====\n{formatted}\n")
                    if args.out:
                        Path(args.out).write_text(formatted + "\n", encoding="utf-8")
                        print(f"[Tersimpan di {args.out}]", file=sys.stderr)
                else:
                    used_model, answer = answer_with_llm(args.query, hits, chat_model=args.chat_model, web_results=web_results)
                    print(f"===== JAWABAN LLM ({used_model}) =====\n{answer}\n")
                    if args.out:
                        payload = {"query": args.query, "model": used_model, "web_results": web_results, "answer": answer}
                        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        print(f"[Tersimpan di {args.out}]", file=sys.stderr)
            except llm.LLMError as e:
                sys.exit(f"Gagal memanggil LLM: {e}")


if __name__ == "__main__":
    main()
