# clear_cache.py
import shutil
import os

def clear_all():
    # ChromaDB (vector store)
    shutil.rmtree("./chroma_db", ignore_errors=True)

    # Python bytecode
    shutil.rmtree("__pycache__", ignore_errors=True)

    print("✅ Cache cleared")

if __name__ == "__main__":
    clear_all()