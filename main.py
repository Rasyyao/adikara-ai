import random
import sys

from chunk import chunk_pages, chunk_stats
from extract import ExtractionError, extract_file


def show_chunk(c: dict) -> str:
    return (f'--- [{c["index"]}] {c["label"]} | {len(c["text"])} karakter ---\n'
            f'{c["text"]}\n')


def inspect(chunks: list[dict], sample: int = 5) -> None:
    # 1. semua chunk ke berkas
    with open("chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(show_chunk(c) for c in chunks))

    # 2. chunk terkecil: biasanya yang paling mencurigakan
    print("===== 3 CHUNK TERKECIL =====")
    for c in sorted(chunks, key=lambda c: len(c["text"]))[:3]:
        print(show_chunk(c))

    # 3. sampel acak
    print("===== SAMPEL ACAK =====")
    random.seed(1)
    for c in random.sample(chunks, min(sample, len(chunks))):
        print(show_chunk(c))

    # 4. perbatasan: akhir satu chunk dan awal chunk berikutnya
    print("===== PERBATASAN (akhir -> awal) =====")
    for a, b in list(zip(chunks, chunks[1:]))[:8]:
        print(f'[{a["index"]}] ...{a["text"][-60:]!r}')
        print(f'[{b["index"]}]    {b["text"][:60]!r}\n')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Pemakaian: python main.py <berkas> [max_chars]")
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    try:
        pages = extract_file(sys.argv[1])
    except ExtractionError as e:
        sys.exit(f"Gagal: {e}")
    chunks = chunk_pages(pages, max_chars=max_chars)
    print(f"{len(pages)} halaman -> {chunk_stats(chunks)}\n")
    inspect(chunks)
    print("\nSemua chunk tersimpan di chunks.txt")