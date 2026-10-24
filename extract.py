import re
import zipfile
from pathlib import Path

from markitdown import MarkItDown

md = MarkItDown()


class ExtractionError(Exception):
    """Pesan aman untuk ditampilkan ke guru."""


SLIDE_MARKER = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->")   # penanda slide dari MarkItDown
TITLE_LINE = re.compile(r"^#\s+(.+)$", re.MULTILINE)                # judul slide: "# Judul"
HEADING = re.compile(r"^#{1,2}\s+(.+?)\s*$")                        # judul DOCX level 1 dan 2

LIST_NUMBER = re.compile(r"^(\d+[.)]|[-•*])\s+")  # "1. ", "- "  -> selalu blok baru
LIST_LETTER = re.compile(r"^[A-Za-z][.)]\s+")     # "A. "       -> hanya jika baris sebelumnya selesai
SENTENCE_END = re.compile(r"[.!?:]\s*$")
PUA_CHARS = re.compile(r"[\uf000-\uf8ff]")
SHORT_LINE = 60   

def normalize_glyphs(text: str) -> str:
    text = re.sub(r"^[ \t]*[\uf000-\uf8ff]+[ \t]*", "- ", text, flags=re.MULTILINE)  # bullet di awal baris
    return PUA_CHARS.sub(" ", text)      

def tidy_text(text: str) -> str:
    """Perapian ringan untuk PPTX dan DOCX: strukturnya sudah benar, jangan diubah."""
    text = re.sub(r"[ \t]+\n", "\n", text)       # spasi di ujung baris
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(text: str) -> str:
    """Khusus PDF: gabungkan baris yang terpotong lebar halaman."""
    text = normalize_glyphs(text)
    text = re.sub(r"[ \t]+", " ", text)           
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and re.fullmatch(r"\d{1,3}", lines[-1]):          # nomor halaman di footer
        lines.pop()

    blocks: list[str] = []
    for line in lines:
        if blocks:
            prev = blocks[-1]
            prev_is_heading = len(prev) <= SHORT_LINE and not SENTENCE_END.search(prev)
            prev_done = bool(SENTENCE_END.search(prev)) or prev_is_heading
            starts_new = bool(
                LIST_NUMBER.match(line)
                or (LIST_LETTER.match(line) and prev_done)   # "gelas\n\nC. Jadi" bukan daftar
                or line.isupper()
                or (SENTENCE_END.search(prev) and line[0].isupper())
                or (prev_is_heading and line[0].isupper())
            )
        else:
            starts_new = True
        if starts_new:
            blocks.append(line)
        else:
            blocks[-1] += " " + line                          # sambung baris yang terpotong
    return "\n\n".join(blocks)


def _convert(path: Path) -> str:
    return md.convert(str(path)).text_content


def extract_pptx(path: Path) -> list[dict]:
    parts = SLIDE_MARKER.split(_convert(path))
    if len(parts) < 3:
        raise ExtractionError("Tidak ada slide yang bisa dibaca dari berkas ini.")
    pages = []
    for number, block in zip(parts[1::2], parts[2::2]):
        match = TITLE_LINE.search(block)
        title = match.group(1).strip() if match else None
        body = TITLE_LINE.sub("", block, count=1) if match else block
        pages.append({"number": int(number), "kind": "slide",
                      "title": title, "text": tidy_text(body)})
    return pages


def extract_pdf(path: Path) -> list[dict]:
    pieces = _convert(path).split("\f")
    while pieces and not pieces[-1].strip():   # buang sisa kosong setelah \f terakhir
        pieces.pop()
    if not any(p.strip() for p in pieces):
        raise ExtractionError("PDF ini tidak berisi teks (mungkin hasil pindaian).")
    return [{"number": i, "kind": "halaman", "title": None, "text": clean_text(p)}
            for i, p in enumerate(pieces, start=1)]


def extract_docx(path: Path) -> list[dict]:
    """DOCX tidak punya halaman tetap, jadi dipecah per judul (# dan ##)."""
    sections: list[dict] = []
    title: str | None = None
    buf: list[str] = []

    def flush():
        body = tidy_text("\n".join(buf))
        if title or body:
            sections.append({"number": len(sections) + 1, "kind": "bagian",
                             "title": title, "text": body})

    for line in _convert(path).splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            title, buf = m.group(1), []
        else:
            buf.append(line)
    flush()
    if not sections:
        raise ExtractionError("Dokumen ini kosong.")
    return sections


EXTRACTORS = {".pptx": extract_pptx, ".pdf": extract_pdf, ".docx": extract_docx}


def extract_file(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ExtractionError(
            f"Jenis berkas '{path.suffix}' belum didukung. Gunakan PPTX, PDF, atau DOCX.")
    if suffix in {".docx", ".pptx"} and not zipfile.is_zipfile(path):
        raise ExtractionError("Berkas bukan dokumen Office yang valid atau rusak.")
    if suffix in {".docx", ".pptx"} and "[Content_Types].xml" not in zipfile.ZipFile(path).namelist():
        raise ExtractionError("Berkas tidak bisa dibaca. Pastikan berkasnya tidak rusak.")
    if suffix == ".pdf" and path.read_bytes()[:5] != b"%PDF-":
        raise ExtractionError("Berkas bukan PDF yang valid atau rusak.")
    try:
        return extractor(path)
    except ExtractionError:
        raise
    except Exception as e:  
        raise ExtractionError("Berkas tidak bisa dibaca. Pastikan berkasnya tidak rusak.") from e


if __name__ == "__main__":
    import json
    import sys

    try:
        print(json.dumps(extract_file(sys.argv[1]), ensure_ascii=False, indent=2))
    except ExtractionError as e:
        sys.exit(f"Gagal: {e}")