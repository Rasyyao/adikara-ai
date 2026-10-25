"""Menyusun pesan untuk model chat: kerangka topik dari materi (RAG) + referensi internet + riwayat percakapan."""
import html
import json
import re
import urllib.parse
import urllib.request

SYSTEM = """Kamu adalah asisten AI pembelajaran yang menggabungkan konteks topik materi guru (RAG) dengan hasil pencarian informasi dari internet (Web Search).
Jawab dalam bahasa Indonesia yang jelas, mendalam, terstruktur, dan mudah dipahami.

Aturan utama penggunaan sumber (RAG + Internet):
1. JANGAN TERLALU FOKUS PADA ISI TEKS PPTX/SLIDE: Bagian "MATERI:" di bawah HANYA berfungsi sebagai penunjuk/peta topik tentang materi apa yang perlu dijelaskan.
2. JANGAN membatasi penjelasan pada kalimat singkat di slide, JANGAN sekadar menyalin contoh sempit dari slide, dan JANGAN mengerjakan soal latihan ("Practice") yang ada di slide kecuali jika diminta secara spesifik oleh pengguna.
3. KOMBINASIKAN RAG DENGAN PENCARIAN INTERNET: Ketika slide tidak menjelaskan secara lengkap (atau hanya menyebut istilah/poin singkat), kamu WAJIB menggunakan informasi dari "REFERENSI INTERNET (WEB SEARCH)" di bawah serta pengetahuan luasmu untuk memberikan penjelasan yang jauh lebih utuh, mendalam, dan aplikatif, tetap dalam kerangka topik materi tersebut.
4. Cantumkan bagian "🌐 Sumber Referensi Internet" di akhir jawaban yang memuat tautan URL dari referensi internet yang digunakan."""


def _chunk_header(index: int, chunk: dict) -> str:
    title = chunk.get("title")
    loc = f"{title} — {chunk['label']}" if title else chunk["label"]
    return f"[{index}] ({loc})"


def _fetch_wiki_summary(lang: str, title: str) -> str | None:
    slug = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "AdikaraAI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.load(resp)
        extract = data.get("extract", "").strip()
        return extract if extract else None
    except Exception:
        return None


def search_web_context(query: str, max_results: int = 5) -> list[dict]:
    """Ambil referensi dari internet (Wikipedia ID/EN & Microsoft Learn) untuk dikombinasikan dengan RAG."""
    cleaned_q = re.sub(r"[^\w\s#+.+-]", " ", query).strip()
    if not cleaned_q:
        return []
    stop_words = {
        "jelaskan", "apa", "itu", "bagaimana", "cara", "dan", "di", "ke", "dari", "yang", "untuk",
        "dengan", "berdasarkan", "materi", "slide", "pptx", "lalu", "perluas", "penjelasan", "tentang",
        "perbedaan", "kapan", "harus", "memakai", "serta", "belum", "ada", "pada", "dalam", "konsep",
    }
    keywords = [w for w in cleaned_q.split() if w.lower() not in stop_words]
    short_q = " ".join((keywords or cleaned_q.split())[:6])
    results: list[dict] = []

    try:
        ms_url = "https://learn.microsoft.com/api/search?" + urllib.parse.urlencode({
            "search": short_q,
            "locale": "id-id",
            "$top": 3,
        })
        req = urllib.request.Request(ms_url, headers={"User-Agent": "AdikaraAI/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            ms_data = json.load(resp)
        for item in ms_data.get("results", []):
            title = item.get("title", "").strip()
            desc = html.unescape(re.sub(r"<[^>]+>", "", item.get("description", "") or "")).strip()
            url = item.get("url", "").strip()
            if title and desc:
                results.append({
                    "title": f"Microsoft Learn: {title}",
                    "url": url,
                    "snippet": desc,
                })
                if len(results) >= max_results:
                    return results
    except Exception:
        pass

    for lang in ("id", "en"):
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": short_q,
                "format": "json",
                "srlimit": 2,
            })
            req = urllib.request.Request(url, headers={"User-Agent": "AdikaraAI/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.load(resp)
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "").strip()
                raw_snip = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", ""))).strip()
                combined_lower = f"{title} {raw_snip}".lower()
                if keywords and not any(k.lower() in combined_lower for k in keywords[:3]):
                    continue
                snippet = _fetch_wiki_summary(lang, title) or raw_snip
                if title and snippet:
                    slug = urllib.parse.quote(title.replace(" ", "_"))
                    results.append({
                        "title": f"Wikipedia ({lang.upper()}): {title}",
                        "url": f"https://{lang}.wikipedia.org/wiki/{slug}",
                        "snippet": snippet,
                    })
                    if len(results) >= max_results:
                        return results
        except Exception:
            continue

    return results[:max_results]


def build_messages(history: list[dict], chunks: list[dict], failed: list[dict], waiting: bool = False,
                   web_results: list[dict] | None = None) -> list[dict]:
    parts = [SYSTEM]
    if chunks:
        parts.append(
            "MATERI: (Gunakan bagian dari slide/dokumen ini HANYA sebagai acuan topik apa yang perlu dijelaskan; "
            "jangan terpaku pada teks singkat atau soal latihan di dalamnya)\n"
            + "\n\n".join(f"{_chunk_header(i, c)}\n{c['text']}" for i, c in enumerate(chunks, 1))
        )
    else:
        parts.append("MATERI: (belum ada materi yang bisa dibaca di percakapan ini)")

    if web_results:
        web_lines = [
            f"- [{w['title']}]({w['url']}): {w['snippet']}" if w.get("url") else f"- {w['title']}: {w['snippet']}"
            for w in web_results
        ]
        parts.append(
            "REFERENSI INTERNET (WEB SEARCH — kombinasikan informasi dari internet ini dengan topik RAG di atas "
            "untuk memberikan penjelasan yang lengkap dan mendalam):\n"
            + "\n".join(web_lines)
        )

    for m in failed:
        parts.append(f"Catatan: berkas \"{m['title']}\" gagal dibaca ({m['error'] or 'alasan tidak diketahui'}). "
                     "Beri tahu guru dengan sopan di awal jawaban dan sarankan mengunggah ulang.")
    return [{"role": "system", "content": "\n\n".join(parts)}, *history]
