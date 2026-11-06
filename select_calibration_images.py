from pathlib import Path
import random
import shutil

src = Path("./calibration_candidates")
dst = Path("./calibration_300")

dst.mkdir(parents=True, exist_ok=True)

files = sorted(src.glob("*.jpg"))

random.seed(42)
selected = random.sample(files, 300)

for file in selected:
    shutil.copy2(file, dst / file.name)

print(f"Selected: {len(selected)} images")
