from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the dataset
DATASET_PDF_PATH = BASE_DIR / "dataset" / "ECMA-262_15th_edition_june_2024-2.pdf"

# Model Name
MODEL_NAME = "llama3.2"