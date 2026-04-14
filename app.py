from flask import Flask, render_template, request, jsonify
from preprocess import preprocess_image, preprocess_text
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-image', methods=['POST'])
def predict_image():
    # Handle Image
    image_file = request.files.get('image')
    image_path = preprocess_image(image_file) if image_file else None
    
    # Handle Text
    raw_text = request.form.get('symptoms')
    processed_text = preprocess_text(raw_text)
    
    # Logic for your model can go here
    # For now, we return a success response
    return jsonify({
        "success": True,
        "message": "Data received and saved.",
        "image_path": image_path,
        "processed_text": processed_text
    })

if __name__ == '__main__':
    app.run(debug=True)
