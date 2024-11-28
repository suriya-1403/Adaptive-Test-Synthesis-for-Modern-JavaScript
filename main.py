import time
from lib import load_and_split_pdf, initialize_components, create_vector_store, query_vector_store
from config import DATASET_PDF_PATH


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
        print(f"Loaded {len(pages)} chunks.")

        # Step 2: Initialize components
        components = initialize_components()
        model = components["model"]
        embeddings = components["embeddings"]
        prompt = components["general_prompt"]
        parser = components["parser"]

        # Step 3: Create vector store
        faiss_index, doc_id_mapping = create_vector_store(pages, embeddings)

        start_time = time.time()

        # Step 4: Query the FAISS vector store
        context_chunks = query_vector_store(faiss_index, doc_id_mapping, embeddings, question, top_k=5)
        context = " ".join(context_chunks)

        # Prepare chain input
        chain_input = {
            "context": context,
            "question": question,
        }

        # Invoke the chain
        chain = prompt | model | parser
        response = chain.invoke(chain_input)

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
