🖼️ Vector Image Search Engine
A high-performance, content-based image retrieval system built with Python, Flask, and FAISS. This engine allows you to find visually similar images from a large database by understanding the "meaning" of an image, not just its pixels.

✨ Screenshots
Here's a look at the application's user interface and search results.

1. Main Interface - Ready to Search

2. Search Results - Similar Images Found
   )

🚀 Features
Content-Based Search: Finds images based on visual similarity (shapes, patterns, concepts).

High Performance: Utilizes Facebook AI's FAISS library for blazingly fast similarity searches.

AI-Powered: Uses the pre-trained CLIP model to generate powerful vector embeddings for images.

Modern UI: A clean and futuristic dark-themed UI built with Flask to upload an image and view results.

Scalable: The architecture is designed to easily handle thousands of new images by simply re-running the indexing script.

🛠️ Tech Stack
Backend: Python, Flask

AI/ML: PyTorch, Sentence-Transformers (for CLIP model)

Vector Search: FAISS (Facebook AI Similarity Search)

Core Libraries: NumPy, Pillow

Frontend: HTML, CSS

⚙️ How It Works
The engine follows a three-step Embed -> Index -> Search process:

Embed: The build_index.py script uses the CLIP model to convert every image in the static/db_images folder into a high-dimensional vector that represents its semantic meaning.

Index: All these vectors are then stored in a highly optimized FAISS index. This index allows for incredibly efficient nearest-neighbor searches.

Search: When a user uploads a new image, it's converted into a vector in real-time. This vector is then used to query the FAISS index to find the most similar images from the database.

🔧 Setup and Installation
To run this project locally, follow these steps:

Clone the repository:

git clone [https://github.com/your-username/VectorSearchEngine.git](https://github.com/your-username/VectorSearchEngine.git)
cd VectorSearchEngine

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install -r requirements.txt

Add your database images:
Place all your images inside the static/db_images/ folder.

Build the search index (one-time step):
This script will process all your images and create the index.faiss file. This may take a few minutes.

python build_index.py

Run the web application:

python app.py

Open your browser and navigate to http://127.0.0.1:5000. Enjoy!
