from flask import Flask, request, jsonify
from models.denoising_model import denoise_audio
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Process the file
    denoised_filepath = denoise_audio(filepath)
    
    return jsonify({'denoised_file': denoised_filepath})

if __name__ == "__main__":
    app.run(debug=True)
