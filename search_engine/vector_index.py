# search_engine/vector_index.py

import os
import faiss
import numpy as np
from tqdm import tqdm
from .embedder import ImageEmbedder

class VectorIndex:
    
    def __init__(self, embedder: ImageEmbedder):
        
        self.embedder = embedder
        self.index = None
        self.image_paths = []

    def build(self, image_folder: str):
       
        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.image_paths:
            print(f"'{image_folder}' no images found.")
            return

        print(f"total {len(self.image_paths)} images found.")

        # Saari images ke liye embeddings (vectors) generate karo
        print("Images are converting into vectors:")
        all_embeddings = []
        for path in tqdm(self.image_paths, desc="Embedding Images"):
            try:
                
                embedding = self.embedder.embed(path)
                if embedding is not None:
                    all_embeddings.append(embedding)
                else:
                    print(f"Warning: '{path}' embedding failed, skipping this image.")
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if not all_embeddings:
            print("no embeddings generated.")
            return
            
      
        embeddings_np = np.array(all_embeddings).astype('float32')
        
      
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        print(f"FAISS index {len(all_embeddings)} vectors added.")

    def save(self, index_path: str, paths_path: str):
        
        if self.index:
            faiss.write_index(self.index, index_path)
            np.save(paths_path, self.image_paths)
            print(f"Index '{index_path}' saved.")
            print(f"Image paths '{paths_path}' saved.")
        else:
            print("no index to save.")

