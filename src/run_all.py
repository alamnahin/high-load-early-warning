"""Run the whole pipeline:  python src/run_all.py   (set RAW_DIR / OUT_DIR env vars if needed)."""
import runpy, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
STEPS = ["01_clean.py", "02_build_features.py", "04_validate.py", "03_eda.py", "05_sanity_baseline.py"]

for s in STEPS:
    t = time.time()
    print(f"\n=== {s} ===")
    runpy.run_path(str(HERE / s), run_name="__main__")
    print(f"--- {s} finished in {time.time() - t:.0f}s")
