from pathlib import Path

import requests
from tqdm import tqdm

def download_with_progress(url: str, dst: Path, chunk_size=1024 * 1024):
    if dst.exists():
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(dst, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dst.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))