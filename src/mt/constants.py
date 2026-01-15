from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
HF_CACHE_DIR = BASE_DIR / ".hfcache"
HF_CACHE_DIR.mkdir(exist_ok=True, parents=True)
