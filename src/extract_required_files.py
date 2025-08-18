# src/extract_required_files.py

import os
import zipfile
import tarfile

base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def extract_zip(zip_path, target_files, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file in target_files:
                zip_ref.extract(file, extract_to)
                print(f"✅ Extracted {file} to {extract_to}")

def extract_tar(tar_path, target_files, extract_to):
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            if os.path.basename(member.name) in target_files:
                tar.extract(member, extract_to)
                print(f"✅ Extracted {member.name} to {extract_to}")

def main():
    # Define what to extract
    extraction_plan = [
        {
            "archive": "EIES_data.zip",
            "target_files": ["eies.csv"],
            "out_dir": "EIES"
        },
        {
            "archive": "fb-forum.zip",
            "target_files": ["fb-forum.edges"],
            "out_dir": "FB_Forum"
        },
        {
            "archive": "soc-epinions-trust-dir.zip",
            "target_files": ["soc-epinions-trust-dir.edges"],
            "out_dir": "Epinions"
        },
        {
            "archive": "download.tsv.dimacs10-celegansneural (1).tar.bz2",
            "target_files": ["out.dimacs10-celegansneural"],
            "out_dir": "Caenorhabditis"
        }
    ]

    for plan in extraction_plan:
        archive_path = os.path.join(base_data_dir, plan["archive"])
        extract_path = os.path.join(base_data_dir, plan["out_dir"])
        os.makedirs(extract_path, exist_ok=True)

        if archive_path.endswith(".zip"):
            extract_zip(archive_path, plan["target_files"], extract_path)
        elif archive_path.endswith((".tar.bz2", ".tar.gz", ".tar")):
            extract_tar(archive_path, plan["target_files"], extract_path)

if __name__ == "__main__":
    main()
