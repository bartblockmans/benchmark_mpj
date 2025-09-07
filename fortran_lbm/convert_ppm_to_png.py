from pathlib import Path
from PIL import Image

src_dir = Path("images")        # change if your PPMs are elsewhere
ppm_files = sorted(src_dir.glob("*.ppm"))

if not ppm_files:
    print("No .ppm files found in", src_dir.resolve())
else:
    for ppm in ppm_files:
        png = ppm.with_suffix(".png")
        with Image.open(ppm) as im:
            im.save(png)  # Pillow auto-detects PNG from extension
        print("Wrote", png)
