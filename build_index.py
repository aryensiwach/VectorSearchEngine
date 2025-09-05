import os
from search_engine.embedder import ImageEmbedder
from search_engine.vector_index import VectorIndex

IMAGE_FOLDER = 'static/db_images'
INDEX_FILE = 'index.faiss'
PATHS_FILE = 'image_paths.npy'

def main():
    print("--- Starting Indexing Process ---")
    
    embedder = ImageEmbedder()
    vector_index = VectorIndex(embedder)
    
    vector_index.build(IMAGE_FOLDER)
    vector_index.save(INDEX_FILE, PATHS_FILE)
    
    print("--- Indexing Process Completed Successfully ---")

if __name__ == "__main__":
    main()

