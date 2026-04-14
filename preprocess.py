import os
import time

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(file):
    """
    Saves the image file to the static/uploads folder.
    Returns the path to the saved image.
    """
    if not file or file.filename == '':
        return None
    
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

def preprocess_text(text):
    """
    Preprocess the symptom/description text.
    Currently just returns the cleaned text.
    """
    if not text:
        return ""
    return text.strip()
