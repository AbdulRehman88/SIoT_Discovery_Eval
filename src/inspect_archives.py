# src/inspect_archives.py

import os
import zipfile
import tarfile

BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def list_zip_contents(zip_path):
    print(f"\n📦 ZIP Archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in zf.namelist():
            print(f"  └── {file}")

def list_tar_contents(tar_path):
    print(f"\n📦 TAR Archive: {tar_path}")
    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf.getmembers():
            print(f"  └── {member.name}")

def main():
    print(f"🔍 Scanning archive files in: {BASE_DATA_DIR}")

    for root, _, files in os.walk(BASE_DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".zip"):
                list_zip_contents(path)
            elif file.endswith((".tar.bz2", ".tar.gz", ".tar")):
                list_tar_contents(path)

if __name__ == "__main__":
    main()
