import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from search_engine.searcher import ImageSearcher

# --- Initial Setup ---
# Flask app ko initialize karein
app = Flask(__name__)

# Uploads ke liye folder set karein
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions set karein
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image Searcher ko load karein (yeh FAISS index aur image paths ko load karega)
# Yeh line server start hote hi ek baar chalegi aur searcher ko ready rakhegi
print("Loading the search index, please wait...")
searcher = ImageSearcher(index_path='index.faiss', paths_path='image_paths.npy')
print("Index loaded successfully!")

# Helper function to check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Web Routes ---

@app.route('/', methods=['GET'])
def index():
    """
    Yeh function main page (index.html) ko render karta hai.
    Jab aap browser mein http://127.0.0.1:5000 kholenge to yeh chalega.
    """
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    Yeh function image search ka kaam handle karta hai.
    Yeh tab chalega jab aap web page se image upload karke "Find Similar Images" button dabayenge.
    """
    # Check karein ki request mein file hai ya nahi
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Agar user ne file select nahi ki hai, to error dein
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Agar file valid hai to search process shuru karein
    if file and allowed_file(file.filename):
        # File ka naam secure karein
        filename = secure_filename(file.filename)
        
        # File ko 'static/uploads' folder mein save karein
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(query_path)

        # ImageSearcher ka use karke similar images dhoondhein
        try:
            # FIXED: 'top_k=10' ko 'k=10' se badal diya gaya hai
            results = searcher.search(query_path, k=10)
            # JSON format mein results wapas bhejein
            return jsonify({'results': results})
        except Exception as e:
            print(f"An error occurred during search: {e}")
            return jsonify({'error': 'An internal error occurred during search'}), 500

    else:
        return jsonify({'error': 'File type not allowed'}), 400

# --- Start the App ---
if __name__ == '__main__':
    # Yeh app ko development mode mein chalata hai
    app.run(host='0.0.0.0', port=5000, debug=True)

