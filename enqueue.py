"""Pengganti sementara rute unggah API: unggah berkas ke R2, buat Material + Job, lalu NOTIFY.

Pakai untuk menguji worker: python enqueue.py data/test-modul.pdf [email-guru]
"""
import json
import os
import sys
import uuid
from pathlib import Path

import storage
import store


def enqueue(path: str, teacher_email: str | None = None) -> str:
    path = Path(path)
    conn = store.connect(autocommit=True)
    if teacher_email:
        row = conn.execute("SELECT id::text FROM teachers WHERE email = %s", (teacher_email,)).fetchone()
    else:
        row = conn.execute("SELECT id::text FROM teachers WHERE role = 'teacher' ORDER BY created_at LIMIT 1").fetchone()
    if row is None:
        sys.exit("Guru tidak ditemukan. Jalankan seed dulu atau beri email guru.")
    key = f"materials/{row[0]}/{uuid.uuid4()}{path.suffix.lower()}"
    storage._client().upload_file(str(path), os.environ["R2_BUCKET"], key)
    material = conn.execute(
        """INSERT INTO materials (id, teacher_id, title, file_path, file_type)
           VALUES (gen_random_uuid(), %s, %s, %s, %s) RETURNING id::text""",
        (row[0], path.stem, key, path.suffix.lower().lstrip("."))).fetchone()[0]
    conn.execute("INSERT INTO jobs (id, type, payload) VALUES (gen_random_uuid(), 'extract', %s)",
                 (json.dumps({"materialId": material}),))
    conn.execute("NOTIFY jobs")      # autocommit: baris sudah commit sebelum NOTIFY
    return material


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Pemakaian: python enqueue.py <berkas> [email-guru]")
    print("material:", enqueue(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
