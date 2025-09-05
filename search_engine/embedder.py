from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

class ImageEmbedder:
    
    def __init__(self):
        try:
            print("🧠 AI model (CLIP) is loading...")
            self.model = SentenceTransformer('clip-ViT-B-32')
            print("✅ AI model safaltapoorvak loaded.")
        except Exception as e:
            print(f"❌ AI model loading error: {e}")
            self.model = None

    def embed(self, image_path):
       
        if not self.model:
            print("❌ Model cant be loaded , embedding isnt possible.")
            return None
            
        try:
           
            image = Image.open(image_path).convert("RGB")
            
            
            embedding = self.model.encode(image)
            
            
            return embedding.tolist()
        except Exception as e:
            print(f"'{image_path}' embedding error: {e}")
            return None

