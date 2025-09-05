import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from search_engine.searcher import ImageSearcher

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

print("Loading the search index, please wait...")
searcher = ImageSearcher(index_path='index.faiss', paths_path='image_paths.npy')
print("Index loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(query_path)

        try:
            results = searcher.search(query_path, k=10)
            return jsonify({'results': results})
        except Exception as e:
            print(f"An error occurred during search: {e}")
            return jsonify({'error': 'An internal error occurred during search'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

