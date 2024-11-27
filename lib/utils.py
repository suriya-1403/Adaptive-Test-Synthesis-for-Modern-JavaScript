import os
from langchain_community.document_loaders import PyPDFLoader
from config import DATASET_PDF_PATH
from log import get_logger

# Initialize the logger
logger = get_logger(__name__)

def load_and_split_pdf(file_path=None):
    """
    Load and split a PDF into pages using PyPDFLoader.

    :param file_path: Optional custom file path to the PDF. Defaults to the dataset PDF path from the config.
    :return: List of pages.
    :raises: FileNotFoundError if the file does not exist.
    """
    # Use the provided file_path or default to DATASET_PDF_PATH
    path_to_pdf = file_path or DATASET_PDF_PATH
    logger.info(f"Using PDF file path: {path_to_pdf}")

    # Ensure the file exists
    if not os.path.exists(path_to_pdf):
        logger.error(f"PDF file not found at path: {path_to_pdf}")
        raise FileNotFoundError(f"PDF file not found at path: {path_to_pdf}")

    try:
        # Load and split the PDF
        logger.info("Starting to load and split the PDF...")
        loader = PyPDFLoader(file_path=str(path_to_pdf))
        pages = loader.load_and_split()
        logger.info(f"Successfully loaded and split the PDF into {len(pages)} pages.")
        return pages
    except Exception as e:
        logger.exception("An error occurred while loading and splitting the PDF.")
        raise
