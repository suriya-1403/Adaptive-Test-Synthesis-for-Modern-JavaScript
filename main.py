from lib import load_and_split_pdf

def main():
    try:
        pages = load_and_split_pdf()
        print(f"Loaded {len(pages)} pages.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()