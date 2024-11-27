from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the dataset
DATASET_PDF_PATH = BASE_DIR / "dataset" / "ECMA-262_15th_edition_june_2024-2.pdf"

# Path to the Vector Store
VECTOR_STORE_PATH = str(BASE_DIR / "storage" /"vector_store")

# Path to the Split Page
SPLIT_PAGES_PATH = BASE_DIR / "storage" /"split_pages.pkl"

# Model Name
# MODEL_NAME = "llama3.2"
MODEL_NAME = "granite3-dense"
# MODEL_NAME = "qwen2.5-coder"
