from PIL import Image
import os

def clean_folder(folder):
    for subdir, _, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                print(f"🧹 Removing: {filepath} ({e})")
                os.remove(filepath)

clean_folder("train")
clean_folder("valid")