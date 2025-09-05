import os
from search_engine.embedder import ImageEmbedder
from search_engine.vector_index import VectorIndex

# --- Configuration ---
# Yeh folder hai jahan aapne apni saari dogs/cats ki images rakhi hain
IMAGE_FOLDER = 'static/db_images'  
# Is naam se search database file save hogi
INDEX_FILE = 'index.faiss'         
# Is naam se image paths ki list save hogi
PATHS_FILE = 'image_paths.npy'     

def main():
    """
    Yeh script aapke image folder se vector index banati hai aur use disk par save karti hai.
    Web application start karne se pehle is script ko ek baar chalana zaroori hai.
    """
    print("--- Index Banane Ki Prakriya Shuru ---")
    
    # 1. Image ko vector mein badalne wala model load karein
    embedder = ImageEmbedder()
    
    # 2. Vector index banane wale tool ko initialize karein
    vector_index = VectorIndex(embedder)
    
    # 3. Apne image folder se index banayein. Yahan time lagega.
    vector_index.build(IMAGE_FOLDER)
    
    # 4. Banaye gaye index aur image paths ki list ko save karein
    # taaki app.py unhe use kar sake
    vector_index.save(INDEX_FILE, PATHS_FILE)
    
    print("--- Index Banane Ki Prakriya Safaltapoorvak Poori Hui ---")

if __name__ == "__main__":
    # Script ko execute karein
    main()

