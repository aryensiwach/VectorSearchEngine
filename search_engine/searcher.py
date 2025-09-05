import faiss
import numpy as np
from .embedder import ImageEmbedder

class ImageSearcher:
    """
    Yeh class FAISS index mein similar images dhoondhti hai.
    """
    def __init__(self, index_path, paths_path):
        self.embedder = ImageEmbedder()
        print(f"'{index_path}' se index load kiya ja raha hai...")
        try:
            self.index = faiss.read_index(index_path)
            # allow_pickle=True zaroori hai NumPy string arrays ko load karne ke liye
            self.image_paths = np.load(paths_path, allow_pickle=True)
            print(f"✅ Index safaltapoorvak load ho gaya. Kul {self.index.ntotal} vectors hain.")
        except Exception as e:
            print(f"❌ Index load karne mein error: {e}")
            self.index = None
            self.image_paths = None

    def search(self, query_image_path, k=10):
        """
        Ek query image ke liye 'k' sabse similar images dhoondhta hai.
        """
        if not self.index:
            print("❌ Index load nahi hua, search nahi ho sakta.")
            return []

        try:
            # 1. Query image ko vector mein badlein
            query_vector = self.embedder.embed(query_image_path)
            
            # Agar embedding fail ho jaaye, to khaali list return karein
            if query_vector is None:
                return []

            # Vector ko FAISS ke liye sahi format (float32 NumPy array) mein badlein
            query_vector_np = np.array([query_vector]).astype('float32')

            # 2. FAISS index mein k-nearest neighbors dhoondhein
            distances, indices = self.index.search(query_vector_np, k)

            # 3. Results (image paths) ko format karein
            # indices[0] mein results ke index numbers hote hain
            results = [self.image_paths[i] for i in indices[0]]
            return results

        except Exception as e:
            # Error ko print karein taaki hume pata chale kya hua
            print(f"❌ Search perform karne mein error: {e}")
            return []

