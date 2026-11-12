"""
Download the real reference cycle from Zenodo.

The cycle is the first cycle of FORinFPRO-HIMD (R. Kini and L. Schweizer, 2026,
doi:10.5281/zenodo.20744054, CC BY 4.0).  Only the two files the benchmark reads are fetched.

    python -m mouldit_benchmark.datasets.download_reference_cycle [--dir DATA_DIR]
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from .reference_preprocessing import DATA_DIR, PT_FILE, US_FILE

RECORD_URL = "https://zenodo.org/api/records/20744054/files/{}/content"
MD5 = {PT_FILE: "40d8511c11e8e0575dc3930ddd258c19", US_FILE: "c767196cfd1b6dec0d09ed0a2dba2551"}


def download(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in MD5.items():
        path = data_dir / name
        if path.exists() and _md5(path) == md5:
            print(f"{path}: present")
            continue
        with urllib.request.urlopen(RECORD_URL.format(name), timeout=60) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != md5:
            raise RuntimeError(f"{name}: checksum mismatch, the Zenodo record may have changed")
        path.write_bytes(content)
        print(f"{path}: downloaded")


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DATA_DIR, help=f"target directory (default: {DATA_DIR})")
    download(ap.parse_args().dir)
