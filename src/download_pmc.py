from pathlib import Path

RAW_DIR = Path("data/raw_pdfs")

def create_folders():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created: {RAW_DIR}")

if __name__ == "__main__":
    create_folders()