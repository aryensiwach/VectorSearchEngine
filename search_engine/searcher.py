import faiss
import numpy as np
from .embedder import ImageEmbedder

class ImageSearcher:
   
    def __init__(self, index_path, paths_path):
        self.embedder = ImageEmbedder()
        print(f"'{index_path}' is loading index...")
        try:
            self.index = faiss.read_index(index_path)
          
            self.image_paths = np.load(paths_path, allow_pickle=True)
            print(f"✅ Index safaltapoorvak loaded. amount {self.index.ntotal} of vectors loaded.")
        except Exception as e:
            print(f"❌ error loading index: {e}")
            self.index = None
            self.image_paths = None

    def search(self, query_image_path, k=10):
        
        if not self.index:
            print("❌ index isnt loaded , cannot search right now.")
            return []

        try:
           
            query_vector = self.embedder.embed(query_image_path)
            
           
            if query_vector is None:
                return []

            
            query_vector_np = np.array([query_vector]).astype('float32')

           
            distances, indices = self.index.search(query_vector_np, k)

            
            results = [self.image_paths[i] for i in indices[0]]
            return results

        except Exception as e:
    
            print(f"❌ error in searching: {e}")
            return []

