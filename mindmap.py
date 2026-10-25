import argparse
import json
import re
import sys

import embed
import llm
import prompt
import store
import worker
from extract import ExtractionError

MINDMAP_SYSTEM = """Kamu adalah asisten pembuat mindmap interaktif bergaya NotebookLM untuk pembelajaran.
Gunakan bagian "MATERI" yang diberikan sebagai ACUAN DASAR (baseline), lalu buat struktur mindmap hierarkis (dari konsep paling umum ke sub-topik hingga detail spesifik dan pengayaan konsep yang relevan dalam konteks topik tersebut) dalam format JSON yang valid.

Aturan wajib:
1. Keluarkan HANYA objek JSON valid tanpa teks tambahan di luar JSON.
2. Struktur JSON wajib memiliki:
   - "title": judul utama mindmap (string)
   - "nodes": daftar objek node (minimal 5 node: 1 node root umum, beberapa node cabang sub-topik dari materi maupun pengembangan konsep yang relevan, dan node detail/contoh)
3. Setiap item di dalam "nodes" WAJIB memiliki field berikut:
   - "id": string unik (misalnya "n1", "n2", "n3")
   - "parent_id": null untuk node utama (root), atau id node induknya (misalnya "n1") untuk cabang/sub-cabang
   - "label": judul singkat node (2-5 kata) yang tampil di kotak mindmap
   - "text": BUKAN rangkuman materi, melainkan KALIMAT PROMPT PERTANYAAN LANJUTAN yang siap dikirim pengguna ke chat saat node ini diklik (seperti fitur klik node mindmap di NotebookLM):
       * Jika node adalah root (paling umum): buat prompt yang meminta penjelasan gambaran umum/menyeluruh tentang topik tersebut berdasarkan materi dan pengembangannya.
       * Jika node adalah cabang (sub-topik): buat prompt yang meminta penjelasan dengan cara yang mudah dipahami tentang sub-topik tersebut dalam kaitannya dengan topik induknya.
       * Jika node adalah detail/contoh spesifik: buat prompt yang meminta penjelasan mendalam dan langkah/contoh konkret mengenai detail tersebut.
   - "source": label halaman/sumber materi tempat konsep dasar ini berada (misalnya "test-modul — Halaman 20-21", atau "Pengayaan Konsep" jika merupakan perluasan relevan di luar slide)"""


def build_mindmap_messages(query: str, chunks: list[dict],
                           web_results: list[dict] | None = None) -> list[dict]:
    parts = [MINDMAP_SYSTEM]
    if chunks:
        parts.append("MATERI: (Gunakan hanya sebagai acuan topik dari slide)\n" + "\n\n".join(
            f"{prompt._chunk_header(i, c)}\n{c['text']}" for i, c in enumerate(chunks, 1)
        ))
    else:
        parts.append("MATERI: (belum ada materi yang dilampirkan; buat mindmap sesuai topik)")
    if web_results:
        web_lines = [
            f"- [{w['title']}]({w['url']}): {w['snippet']}" if w.get("url") else f"- {w['title']}: {w['snippet']}"
            for w in web_results
        ]
        parts.append("REFERENSI INTERNET (kombinasikan dengan topik materi di atas):\n" + "\n".join(web_lines))
    user_prompt = (
        f"Buat mindmap JSON ala NotebookLM untuk topik/materi berikut: {query}\n"
        "Ingat: field 'text' pada setiap node WAJIB berupa kalimat prompt pertanyaan lanjutan "
        "(mulai dari yang bersifat umum di node root hingga penjelasan mudah/detail di node cabang), "
        "bukan rangkuman jawaban."
    )
    return [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": user_prompt},
    ]


def parse_mindmap_json(raw: str, fallback_title: str = "Mindmap") -> dict:
    """Bersihkan, validasi, dan normalisasi JSON mindmap agar setiap node pasti punya `text` (prompt klik)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise llm.LLMError(f"LLM tidak mengembalikan JSON mindmap yang valid: {e}") from e

    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list) or not data["nodes"]:
        raise llm.LLMError("JSON mindmap harus berupa objek dengan daftar 'nodes' yang tidak kosong.")

    normalized_nodes = []
    edges = []
    seen_ids = set()

    for idx, node in enumerate(data["nodes"], start=1):
        if not isinstance(node, dict):
            continue
        node_data = node.get("data") if isinstance(node.get("data"), dict) else {}
        node_id = str(node.get("id") or f"n{idx}").strip()
        seen_ids.add(node_id)
        parent_id = node.get("parent_id", node.get("parentId"))
        parent_id = str(parent_id).strip() if parent_id else None

        label = str(node.get("label") or node_data.get("label") or f"Konsep {idx}").strip()
        text = str(
            node.get("text")
            or node.get("prompt")
            or node_data.get("text")
            or node_data.get("prompt")
            or ""
        ).strip()
        source = str(node.get("source") or node_data.get("source") or "").strip() or None

        if not text:
            raise llm.LLMError(f"Node '{node_id}' ({label}) tidak memiliki data 'text' (prompt).")

        normalized_nodes.append({
            "id": node_id,
            "parent_id": parent_id,
            "label": label,
            "text": text,
            "source": source,
            "data": {
                "label": label,
                "text": text,
                "source": source,
            },
        })

    if not normalized_nodes:
        raise llm.LLMError("JSON mindmap tidak memiliki node yang valid.")

    for node in normalized_nodes:
        pid = node["parent_id"]
        if pid and pid in seen_ids:
            edges.append({
                "id": f"e-{pid}-{node['id']}",
                "source": pid,
                "target": node["id"],
            })

    return {
        "title": str(data.get("title") or fallback_title).strip(),
        "nodes": normalized_nodes,
        "edges": edges,
    }


def generate_mindmap(query: str, chunks: list[dict], model: str | None = None,
                     endpoint: str | None = None,
                     web_results: list[dict] | None = None) -> dict:
    default_model, default_endpoint = worker._chat_config()
    used_model = llm.resolve_model(model or default_model or llm.DEFAULT_CHAT_MODEL)
    used_endpoint = endpoint or (default_endpoint if not model else "local")
    if web_results is None:
        web_results = prompt.search_web_context(query)
    messages = build_mindmap_messages(query, chunks, web_results=web_results)
    raw = llm.chat(used_model, messages, endpoint=used_endpoint, json_mode=True)
    result = parse_mindmap_json(raw, fallback_title=query)
    result["model"] = used_model
    return result


def main(argv: list[str] | None = None) -> None:
    import search

    parser = argparse.ArgumentParser(
        description="Embed topik/pertanyaan -> cari top-k di pgvector -> hasilkan JSON mindmap ala NotebookLM."
    )
    parser.add_argument("query", nargs="?", default=search.DEFAULT_QUERY,
                        help=f"topik atau pertanyaan mindmap (bawaan: '{search.DEFAULT_QUERY}')")
    parser.add_argument("--file", help="berkas lokal (PDF/PPTX/DOCX) untuk diindeks langsung ke Postgres")
    parser.add_argument("--material", help="UUID materi di Postgres")
    parser.add_argument("--chat", help="UUID chat di Postgres")
    parser.add_argument("--k", type=int, default=6, help="jumlah top-k chunk materi (bawaan: 6)")
    parser.add_argument("--min-score", type=float, default=0.0, help="batas skor kemiripan minimum (bawaan: 0.0)")
    parser.add_argument("--model", default=embed.FINAL_MODEL, help=f"model embedding (bawaan: {embed.FINAL_MODEL})")
    parser.add_argument("--endpoint", default="local", help="endpoint Ollama untuk embedding (bawaan: local)")
    parser.add_argument("--chat-model", help=f"model LLM (bawaan: {llm.DEFAULT_CHAT_MODEL})")
    parser.add_argument("--chat-models", nargs="+", help="daftar model LLM untuk dibandingkan secara berdampingan")
    parser.add_argument("--compare", action="store_true",
                        help=f"bandingkan 3 model bawaan: {', '.join(llm.COMPARE_MODELS)}")
    parser.add_argument("--out", help="simpan hasil JSON mindmap ke berkas (opsional, mis. mindmap.json)")
    args = parser.parse_args(argv)

    if args.k < 1:
        sys.exit("Gagal: --k harus bilangan bulat positif.")

    try:
        with store.connect(autocommit=True) as conn:
            hits, target = search.run_search(
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
        if args.compare or args.chat_models:
            models_to_run = args.chat_models if args.chat_models else list(llm.COMPARE_MODELS)
            comp_results = search.compare_with_llm(args.query, hits, models=models_to_run, mindmap_mode=True)
            print(search.format_comparison(comp_results, mindmap_mode=True))
            if args.out:
                payload = {"query": args.query, "target": target, "k": args.k, "results": comp_results}
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                print(f"\n[Tersimpan di {args.out}]", file=sys.stderr)
            if all(r["status"] != "ok" for r in comp_results):
                sys.exit("Gagal: semua model LLM gagal merespons.")
            return
        mindmap_data = generate_mindmap(args.query, hits, model=args.chat_model)
    except (store.StoreError, embed.EmbedError, ExtractionError, llm.LLMError) as e:
        sys.exit(f"Gagal: {e}")

    formatted = json.dumps(mindmap_data, ensure_ascii=False, indent=2)
    print(formatted)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(formatted + "\n")
        print(f"\n[Tersimpan di {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
