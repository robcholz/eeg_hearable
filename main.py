from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Fhrozen/FSD50k"
DEV = "dev"
EVAL = "eval"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=OUTPUT_DIR,
                  allow_patterns=[f"{DEV}/**", f"{EVAL}/**"], )
print(f"Downloaded {DEV} and {EVAL} to {OUTPUT_DIR.resolve()}")
