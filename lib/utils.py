import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DATASET_PDF_PATH, MODEL_NAME
from log import get_logger
from pydantic import ValidationError
import faiss
import numpy as np

# Initialize the logger
logger = get_logger(__name__)


def load_and_split_pdf(file_path=None, chunk_size=500, chunk_overlap=50):
    """
    Load and split a PDF into smaller chunks using RecursiveCharacterTextSplitter.

    :param file_path: str, optional
        Custom file path to the PDF. Defaults to the dataset PDF path from the config.
    :param chunk_size: int
        Maximum size of each chunk.
    :param chunk_overlap: int
        Overlap between chunks to maintain context.
    :return: list
        List of chunks extracted from the PDF.
    :raises FileNotFoundError:
        If the file does not exist.
    :raises Exception:
        If any other error occurs during the PDF loading and splitting process.
    """
    # Use the provided file_path or default to DATASET_PDF_PATH
    path_to_pdf = file_path or DATASET_PDF_PATH
    logger.info(f"Using PDF file path: {path_to_pdf}")

    # Ensure the file exists
    if not os.path.exists(path_to_pdf):
        logger.error(f"PDF file not found at path: {path_to_pdf}")
        raise FileNotFoundError(f"PDF file not found at path: {path_to_pdf}")

    try:
        # Load the PDF
        logger.info("Starting to load the PDF...")
        loader = PyPDFLoader(file_path=str(path_to_pdf))
        pages = loader.load()

        logger.info(f"Loaded {len(pages)} pages from the PDF.")

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Split each page into smaller chunks
        logger.info("Splitting pages into smaller chunks...")
        chunks = []
        for page in pages:
            chunks.extend(text_splitter.split_text(page.page_content))

        logger.info(f"Successfully split the PDF into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.exception("An error occurred while loading and splitting the PDF.")
        raise


def initialize_components():
    """
    Initialize the LangChain components: model, embeddings, parser, and prompt templates.

    :return: dict
        A dictionary containing the initialized components:
        - **model** (Ollama): The LLM model.
        - **embeddings** (OllamaEmbeddings): The embedding model.
        - **parser** (StrOutputParser): The output parser.
        - **general_prompt** (PromptTemplate): The general-purpose prompt template.
        """
    try:
        logger.info("Initializing LangChain components...")

        # Initialize the LLM model
        model = OllamaLLM(model=MODEL_NAME)
        logger.info(f"LLM model '{MODEL_NAME}' initialized.")

        # Initialize the embeddings
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        logger.info("Embeddings initialized.")

        # Initialize the output parser
        parser = StrOutputParser()
        logger.info("Output parser initialized.")

        # Initialize the general-purpose prompt template
        general_prompt_template = """
        You are a JavaScript testing expert. Your task is to generate test cases for the given function based on its specification.

        Please ensure the following while generating the test cases:
        1. Use the provided context as the authoritative specification for the function's behavior.
        2. Generate test cases that cover:
           - Typical use cases.
           - Edge cases, including boundary conditions and special values.
           - Invalid inputs, ensuring proper error handling, if applicable.
        3. The output should only include JavaScript code with `assert` statements. Do not include comments or explanations.
        4. Ensure the test cases are valid and adhere to the provided specification.

        Context: {context}

        Question: {question}
        """
        general_prompt = PromptTemplate(template=general_prompt_template)
        logger.info("General-purpose prompt template initialized.")

        return {
            "model": model,
            "embeddings": embeddings,
            "parser": parser,
            "general_prompt": general_prompt,
        }
    except Exception as error:
        logger.exception("Error during component initialization.")
        raise error


def create_vector_store(chunks, embeddings, nlist=100):
    """
    Create a FAISS vector store from the provided chunks using an Inverted File Index (IVF).

    :param chunks: list
        List of text chunks extracted from the PDF.
    :param embeddings: OllamaEmbeddings
        Embedding model for vectorization.
    :param nlist: int, optional
        Number of clusters for the Inverted File Index. Defaults to 100.
    :return: tuple
        A tuple containing:
        - The FAISS IVF index (faiss.IndexIVFFlat).
        - A mapping of document IDs to their corresponding chunks.
    :raises Exception:
        If any error occurs during vector store creation.
    """
    try:
        logger.info("Creating FAISS vector store using IVF...")

        # Use embed_query to generate embeddings
        embedding_vectors = np.array([embeddings.embed_query(chunk) for chunk in chunks]).astype('float32')

        # Determine the dimensionality of the embeddings
        dimension = embedding_vectors.shape[1]

        # Initialize the quantizer (Flat index for clustering)
        quantizer = faiss.IndexFlatL2(dimension)  # Base index for clustering
        logger.info(f"Quantizer initialized with dimension {dimension}.")

        # Create an IVF index
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)  # IVF index with `nlist` clusters

        # Train the index (mandatory for IVF)
        index.train(embedding_vectors)
        logger.info(f"IVF index trained with {nlist} clusters.")

        # Add the embeddings to the index
        index.add(embedding_vectors)
        logger.info("Embeddings added to the FAISS IVF index.")

        # Create a mapping of document IDs to their content
        doc_id_mapping = {i: chunk for i, chunk in enumerate(chunks)}
        logger.info("Document ID mapping created.")

        return index, doc_id_mapping
    except Exception as error:
        logger.exception("An error occurred while creating the FAISS vector store using IVF.")
        raise error





def query_vector_store(index, doc_id_mapping, embeddings, query, top_k=5):
    """
    Query the FAISS vector store to retrieve the most relevant chunks.

    :param index: faiss.IndexIVFFlat
        The FAISS IVF index.
    :param doc_id_mapping: dict
        A mapping of document IDs to their content.
    :param embeddings: OllamaEmbeddings
        The embedding model.
    :param query: str
        The query text.
    :param top_k: int, optional
        Number of top results to retrieve. Defaults to 5.
    :return: list
        List of the most relevant chunks.
    """
    # Generate embedding for the query
    query_vector = np.array([embeddings.embed_query(query)]).astype('float32')

    # Perform the search
    distances, indices = index.search(query_vector, top_k)

    # Map results back to document chunks
    results = [doc_id_mapping[idx] for idx in indices[0]]
    return results

