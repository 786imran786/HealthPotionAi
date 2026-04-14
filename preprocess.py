import os
import base64
from PIL import Image
import io
import time

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def validate_image(file):
    """
    Validate the uploaded image file.
    """
    if not file or file.filename == '':
        return False, "No file selected."
    
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, ""

def preprocess_image(file):
    """
    Preprocess the image:
    1. Save it to the static/uploads folder.
    2. Convert to base64 for Gemini API.
    """
    # Generate unique filename to avoid collisions
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Reset file pointer to beginning before saving/reading
    file.seek(0)
    
    # Save original for reference
    file.save(filepath)
    
    # Re-open it using PIL to process (resize if it's too large for API, though Gemini handles large images)
    # We'll just read it for base64 conversion
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    mime_type = file.content_type
    if not mime_type:
        ext = filename.rsplit('.', 1)[1].lower()
        mime_type = f"image/{ext}"
        if ext == 'jpg': mime_type = "image/jpeg"

    return encoded_string, mime_type
