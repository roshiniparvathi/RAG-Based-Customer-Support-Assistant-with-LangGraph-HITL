try:
    import langchain_huggingface
    print("langchain_huggingface imported successfully")
    print(f"Version: {langchain_huggingface.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
