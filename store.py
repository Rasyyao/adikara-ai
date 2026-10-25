"""Memori RAG per chat di Postgres (pgvector).

Chunk disimpan per materi (sekali embed). Sebuah chat hanya membaca materi yang dilampirkan ke chat itu
(chat_materials), jadi chat satu tidak pernah melihat materi chat lain.
"""
import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

import embed
from chunk import chunk_pages
from extract import extract_file

embed.load_env()


class StoreError(Exception):
    """Pesan aman untuk ditampilkan ke pengguna."""


def connect(autocommit: bool = False) -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise StoreError("DATABASE_URL belum diatur (lihat .env.example).")
    conn = psycopg.connect(url, autocommit=autocommit)
    register_vector(conn)
    return conn


def index_material(conn: psycopg.Connection, material_id: str, path: str | Path,
                   model: str = embed.FINAL_MODEL, endpoint: str = "local") -> int:
    """Ekstrak, pecah, embed, lalu simpan chunk materi sekali saja. Memanggil ulang menggantikan hasil model ini."""
    chunks = chunk_pages(extract_file(path))
    if not chunks:
        raise StoreError("Materi tidak menghasilkan potongan teks.")
    vectors = embed.embed_documents(model, [c["embed_text"] for c in chunks], endpoint)
    with conn.transaction():
        conn.execute("DELETE FROM material_chunks WHERE material_id = %s AND embedding_model = %s",
                     (material_id, model))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO material_chunks
                   (id, material_id, chunk_index, label, text, embedding, embedding_model)
                   VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s)""",
                [(material_id, c["index"], c["label"], c["text"],
                  np.asarray(v, dtype=np.float32), model) for c, v in zip(chunks, vectors)],
            )
    return len(chunks)


def _rows_to_chunks(rows: list[tuple]) -> list[dict]:
    return [
        {"id": r[0], "title": r[1], "label": r[2], "text": r[3], "score": float(r[4])}
        for r in rows
    ]


def retrieve(conn: psycopg.Connection, chat_id: str, query: str, k: int = 4,
             model: str = embed.FINAL_MODEL, endpoint: str = "local",
             min_score: float = 0.0) -> list[dict]:
    """k chunk terdekat, hanya dari materi yang dilampirkan ke chat ini (chat_materials).
    Model harus sama dengan model saat indexing."""
    vec = np.asarray(embed.embed_query(model, query, endpoint), dtype=np.float32)
    rows = conn.execute(
        """SELECT mc.id::text, m.title, mc.label, mc.text, 1 - (mc.embedding <=> %s) AS score
           FROM chat_materials cm
           JOIN materials m ON m.id = cm.material_id
           JOIN material_chunks mc ON mc.material_id = cm.material_id
           WHERE cm.chat_id = %s AND mc.embedding_model = %s
             AND 1 - (mc.embedding <=> %s) >= %s
           ORDER BY mc.embedding <=> %s
           LIMIT %s""",
        (vec, chat_id, model, vec, min_score, vec, k),
    ).fetchall()
    return _rows_to_chunks(rows)


def retrieve_material(conn: psycopg.Connection, material_id: str, query: str, k: int = 4,
                      model: str = embed.FINAL_MODEL, endpoint: str = "local",
                      min_score: float = 0.0) -> list[dict]:
    """k chunk terdekat langsung dari satu materi (tanpa perlu chat_materials)."""
    vec = np.asarray(embed.embed_query(model, query, endpoint), dtype=np.float32)
    rows = conn.execute(
        """SELECT mc.id::text, m.title, mc.label, mc.text, 1 - (mc.embedding <=> %s) AS score
           FROM material_chunks mc
           JOIN materials m ON m.id = mc.material_id
           WHERE mc.material_id = %s AND mc.embedding_model = %s
             AND 1 - (mc.embedding <=> %s) >= %s
           ORDER BY mc.embedding <=> %s
           LIMIT %s""",
        (vec, material_id, model, vec, min_score, vec, k),
    ).fetchall()
    return _rows_to_chunks(rows)
