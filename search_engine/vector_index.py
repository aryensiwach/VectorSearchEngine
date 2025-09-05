import os
import faiss
import numpy as np
from tqdm import tqdm

class VectorIndex:
    """
    Yeh class image vectors ka ek searchable index (library) banati aur manage karti hai.
    Yeh FAISS library ka use karti hai jo bahut fast similarity search karti hai.
    """
    def __init__(self, embedder):
        """
        Index ko initialize karta hai.
        
        Args:
            embedder (ImageEmbedder): Image ko vector mein badalne wala object.
        """
        self.embedder = embedder
        self.index = None
        self.image_paths = []

    def build(self, image_folder):
        """
        Diye gaye folder se saari images ko process karke ek naya FAISS index banata hai.
        
        Args:
            image_folder (str): Us folder ka path jahan saari database images rakhi hain.
        """
        print(f"'{image_folder}' se index banaya ja raha hai...")
        
        # Folder mein saari valid image files (.jpg, .png, etc.) ko dhoondho
        all_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Kul {len(all_files)} images mili.")

        # Saari images ke vectors nikaalo
        # tqdm ek progress bar dikhata hai taaki pata chale kaam kitna hua
        embeddings = []
        for path in tqdm(all_files, desc="Images ko vectors mein badla ja raha hai"):
            embedding = self.embedder.get_embedding(path)
            if embedding is not None:
                embeddings.append(embedding)
                self.image_paths.append(path)
        
        if not embeddings:
            print("❌ Koi bhi image process nahi ho payi. Folder check karein.")
            return

        # Vectors ko FAISS ke liye sahi format (numpy array) mein convert karo
        embeddings = np.array(embeddings).astype('float32')
        
        # FAISS index banayein
        dimension = embeddings.shape[1]  # Vector ka size (e.g., 512)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✅ Index safaltapoorvak ban gaya. Kul {self.index.ntotal} vectors add kiye gaye.")

    def save(self, index_path, paths_path):
        """
        Banaye gaye index aur image paths ko disk par save karta hai.
        
        Args:
            index_path (str): FAISS index file ko save karne ka path.
            paths_path (str): Image paths ki list ko save karne ka path.
        """
        print(f"Index ko '{index_path}' par save kiya ja raha hai...")
        faiss.write_index(self.index, index_path)
        
        print(f"Image paths ko '{paths_path}' par save kiya ja raha hai...")
        np.save(paths_path, np.array(self.image_paths, dtype=object))
        
        print("✅ Index aur paths safaltapoorvak save ho gaye.")

