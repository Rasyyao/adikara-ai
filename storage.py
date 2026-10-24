"""Unduh berkas materi dari Cloudflare R2 (kompatibel S3). Kunci objek disimpan di materials.file_path."""
import os
import tempfile
from pathlib import Path

import embed

embed.load_env()


class StorageError(Exception):
    """Pesan aman untuk ditampilkan ke pengguna."""


def _client():
    import boto3
    from botocore.config import Config

    missing = [k for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
               if not os.environ.get(k)]
    if missing:
        raise StorageError(f"Penyimpanan belum diatur: {', '.join(missing)} (lihat .env.example).")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 3}, connect_timeout=10, read_timeout=60),
    )


def download(key: str, dest_dir: str | Path) -> Path:
    """Simpan objek `key` ke dest_dir dengan ekstensi aslinya (extract_file memilih pembaca dari ekstensi)."""
    dest = Path(dest_dir) / f"material{Path(key).suffix.lower()}"
    try:
        _client().download_file(os.environ["R2_BUCKET"], key, str(dest))
    except StorageError:
        raise
    except Exception as e:
        raise StorageError("Berkas tidak ditemukan di penyimpanan.") from e
    return dest


def fetch(key: str) -> tempfile.TemporaryDirectory:
    """Unduh ke folder sementara; panggil .cleanup() atau pakai sebagai context manager."""
    tmp = tempfile.TemporaryDirectory(prefix="adikara-")
    try:
        download(key, tmp.name)
    except Exception:
        tmp.cleanup()
        raise
    return tmp
