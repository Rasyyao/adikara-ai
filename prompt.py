"""Menyusun pesan untuk model chat: aturan asisten, konteks dari materi chat ini, lalu riwayat percakapan."""

SYSTEM = """Kamu adalah asisten untuk guru yang menyiapkan pembelajaran di kelas.
Bantu guru memahami materi, menyusun penjelasan, contoh, soal, dan kegiatan kelas.
Jawab dalam bahasa Indonesia yang jelas dan ringkas.
Pakai bagian "MATERI" di bawah sebagai sumber utama. Jika jawabannya tidak ada di materi, katakan terus terang \
bahwa materi tidak membahasnya, lalu beri saran umum yang ditandai sebagai saran umum, bukan isi materi."""


def build_messages(history: list[dict], chunks: list[dict], failed: list[dict], waiting: bool = False) -> list[dict]:
    """history: [{"role": "user"|"assistant", "content": str}] dari lama ke baru, diakhiri pesan guru."""
    parts = [SYSTEM]
    if chunks:
        parts.append("MATERI:\n" + "\n\n".join(f"[{i}] ({c['label']})\n{c['text']}" for i, c in enumerate(chunks, 1)))
    else:
        parts.append("MATERI: (belum ada materi yang bisa dibaca di percakapan ini)")
    for m in failed:
        parts.append(f"Catatan: berkas \"{m['title']}\" gagal dibaca ({m['error'] or 'alasan tidak diketahui'}). "
                     "Beri tahu guru dengan sopan di awal jawaban dan sarankan mengunggah ulang.")
    return [{"role": "system", "content": "\n\n".join(parts)}, *history]
