from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

class ImageEmbedder:
    """
    Yeh class ek image ko high-dimensional vector mein badalti hai.
    """
    def __init__(self):
        # Pre-trained CLIP model ko load karein. Yeh model images ke "meaning" ko samajhta hai.
        # Yeh line model ko download karegi agar pehli baar run ho raha hai.
        try:
            print("🧠 AI model (CLIP) load ho raha hai...")
            self.model = SentenceTransformer('clip-ViT-B-32')
            print("✅ AI model safaltapoorvak load ho gaya.")
        except Exception as e:
            print(f"❌ AI model load karne mein error: {e}")
            self.model = None

    def embed(self, image_path):
        """
        Ek image file path leta hai aur uska embedding (vector) return karta hai.
        """
        if not self.model:
            print("❌ Model load nahi hua, embedding nahi ho sakta.")
            return None
            
        try:
            # Image ko kholein aur use standard RGB format mein convert karein
            image = Image.open(image_path).convert("RGB")
            
            # Model ka use karke image ko encode karein. Yeh ek NumPy array return karega.
            embedding = self.model.encode(image)
            
            # Embedding ko ek standard Python list (of floats) mein convert karke return karein
            return embedding.tolist()
        except Exception as e:
            print(f"'{image_path}' ko embed karne mein error: {e}")
            return None

