import time
from lib import load_and_split_pdf, get_cached_components, create_chroma_vector_store
from config import DATASET_PDF_PATH
from log import get_logger
from pydantic import ValidationError

# Initialize the logger
logger = get_logger(__name__)

def main(question):
    """
    Main function to handle the LangChain workflow for a single question.

    :param question: str
        The question to ask the LLM using the context retrieved from the vector store.
    :raises FileNotFoundError:
        If the PDF file to load is not found.
    :raises Exception:
        If any other error occurs during the process.
    """
    try:
        # Step 1: Load and split PDF
        pages = load_and_split_pdf(DATASET_PDF_PATH)

        # Step 2: Initialize components
        components = get_cached_components()
        model = components["model"]
        embeddings = components["embeddings"]
        prompt = components["general_prompt"]
        parser = components["parser"]

        # Step 3: Create vector store
        vectorstore = create_chroma_vector_store(pages, embeddings)

        # Step 4: Query the vector store and invoke the chain

        retriever = vectorstore.as_retriever()

        start_time = time.time()

        # Retrieve relevant documents
        logger.info("Retrieving context from vector store...")
        # context_docs = retriever.invoke(question)
        context_docs = retriever.get_relevant_documents(question)
        if not context_docs:
            logger.error("No relevant context found in the vector store.")
            return
        context = " ".join([doc.page_content for doc in context_docs])
        logger.info(f"Retrieved context: ...")
        print(context)
        # Prepare chain input
        logger.info("Initializing Chain...")
        chain_input = {
            "context": context,
            "question": question,
        }


        # Invoke the chain
        chain = prompt | model | parser
        logger.info("Chain initialized successfully.")

        logger.info("Generating test cases...")
        response = chain.invoke(chain_input)
        logger.info("Test cases generated successfully.")
        end_time = time.time()

        # Print results
        print(f"Question: {question}")
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("-" * 30)

    except FileNotFoundError as error:
        print(error)
    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    question_to_ask = "Generate test cases for the function: Math.abs()"
    main(question_to_ask)
