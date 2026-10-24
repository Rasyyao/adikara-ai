import re

from extract import SENTENCE_END, SHORT_LINE

DEFAULT_MAX_CHARS = 1000

# satu unit = beberapa blok yang tidak boleh dipisah; tiap blok membawa nomor halamannya
Block = tuple[int, str]


def _is_open(block: str, next_block: str | None) -> bool:
    """Blok ini 'menggantung' dan harus ikut blok sesudahnya."""
    if block.rstrip().endswith(":"):
        return True                                            # kalimat pengantar
    if len(block) <= SHORT_LINE and not SENTENCE_END.search(block):
        return True                                            # judul pendek
    if next_block and not SENTENCE_END.search(block) and next_block[0].islower():
        return True                                            # kalimat bersambung ke blok berikutnya
    return False


def _group(blocks: list[Block]) -> list[list[Block]]:
    units: list[list[Block]] = []
    current: list[Block] = []
    for i, (page, text) in enumerate(blocks):
        current.append((page, text))
        nxt = blocks[i + 1][1] if i + 1 < len(blocks) else None
        if not _is_open(text, nxt):
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def _split_oversized(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        cut = cut + 1 if cut > 0 else window.rfind(" ")
        cut = cut if cut > 0 else max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _pack(units: list[list[Block]], max_chars: int) -> list[dict]:
    """Isi chunk sampai hampir penuh. Batas halaman bersifat lunak: chunk baru dimulai
    di halaman baru, kecuali sebuah unit memang menyeberang halaman."""
    out: list[dict] = []
    cur_text, cur_pages = "", []

    def flush():
        nonlocal cur_text, cur_pages
        if cur_text:
            out.append({"text": cur_text, "start": min(cur_pages), "end": max(cur_pages)})
        cur_text, cur_pages = "", []

    for unit in units:
        pages = [p for p, _ in unit]
        text = "\n\n".join(t for _, t in unit)
        for part in _split_oversized(text, max_chars):
            new_page = bool(cur_pages) and min(pages) > max(cur_pages)
            candidate = f"{cur_text}\n\n{part}" if cur_text else part
            if cur_text and (len(candidate) > max_chars or new_page):
                flush()
                candidate = part
            cur_text = candidate
            cur_pages += pages
    flush()
    return out


def _label(kind: str, start: int, end: int) -> str:
    name = kind.capitalize()
    return f"{name} {start}" if start == end else f"{name} {start}-{end}"


def chunk_pages(pages: list[dict], max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    chunks: list[dict] = []
    # halaman PDF adalah potongan kertas, bukan satuan makna: kalimat bisa menyeberang.
    # slide dan bagian DOCX adalah satuan makna: diproses sendiri-sendiri.
    streams: list[list[dict]] = []
    for page in pages:
        if page["kind"] == "halaman" and streams and streams[-1][0]["kind"] == "halaman":
            streams[-1].append(page)
        else:
            streams.append([page])

    for stream in streams:
        blocks: list[Block] = []
        for page in stream:
            for b in page["text"].split("\n\n"):
                if b.strip():
                    blocks.append((page["number"], b.strip()))
        if not blocks:
            continue
        for piece in _pack(_group(blocks), max_chars):
            label = _label(stream[0]["kind"], piece["start"], piece["end"])
            title = stream[0]["title"] if len(stream) == 1 else None
            header = f"{label}: {title}" if title else label
            chunks.append({
                "index": len(chunks),
                "page_number": piece["start"],
                "page_start": piece["start"],
                "page_end": piece["end"],
                "kind": stream[0]["kind"],
                "title": title,
                "label": label,
                "text": piece["text"],
                "embed_text": f"{header}\n{piece['text']}",
            })
    return chunks


def chunk_stats(chunks: list[dict]) -> str:
    lengths = sorted(len(c["text"]) for c in chunks)
    if not lengths:
        return "tidak ada chunk"
    small = sum(1 for n in lengths if n < 100)
    return (f"{len(lengths)} chunk | min {lengths[0]} | median {lengths[len(lengths)//2]} "
            f"| maks {lengths[-1]} | di bawah 100 karakter: {small}")