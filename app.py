"""
app.py — Flask backend for HealthPotion AI.

Serves the web interface and processes health analysis requests
via the Google Gemini API (image and text-based analysis).
"""

import os
import json
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from preprocess import validate_image, preprocess_image

# ─── App Configuration ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure upload directory exists
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

# ─── Gemini AI Configuration ────────────────────────────────────────────────
# Set your API key as an environment variable: GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model to use (gemini-2.0-flash is fast and multimodal)
MODEL_NAME = "gemini-2.0-flash"

# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are HealthPotion AI, an advanced medical health assistant.
Your role is to provide helpful, informative health insights based on the user's input.

IMPORTANT GUIDELINES:
- Always include a clear DISCLAIMER that you are an AI and not a licensed medical professional.
- Recommend consulting a real healthcare provider for proper diagnosis and treatment.
- Provide well-structured, easy-to-understand responses.
- If analyzing an image, describe what you observe and potential health-related insights.
- If analyzing symptoms, provide possible conditions, general advice, and when to seek medical help.
- Be empathetic and supportive in your tone.
- Format your response in clear sections using markdown.

RESPONSE FORMAT:
## 🔍 Analysis Summary
Brief overview of findings.

## 📋 Detailed Observations
- Bullet points of key observations.

## 💡 Recommendations
- Actionable suggestions and next steps.

## ⚠️ Disclaimer
AI-generated insights — always consult a qualified healthcare professional for medical advice.
"""


def get_gemini_model():
    """Get a configured Gemini model instance."""
    if not GEMINI_API_KEY:
        return None
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main homepage."""
    return render_template("index.html")


@app.route("/analyze/image", methods=["POST"])
def analyze_image():
    """
    Analyze an uploaded medical image using Gemini Vision.

    Expects: multipart/form-data with an 'image' file field.
    Returns: JSON with analysis results.
    """
    try:
        # Check API key
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key not configured. Set the GEMINI_API_KEY environment variable."
            }), 500

        # Validate file presence
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided."}), 400

        file = request.files["image"]

        # Validate image
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({"success": False, "error": error_msg}), 400

        # Preprocess image
        base64_data, mime_type = preprocess_image(file)

        # Build Gemini request
        model = get_gemini_model()
        image_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_data,
            }
        }

        prompt = (
            "Please analyze this medical/health-related image. "
            "Describe what you see, identify any potential health concerns, "
            "and provide helpful insights and recommendations."
        )

        response = model.generate_content([prompt, image_part])

        return jsonify({
            "success": True,
            "analysis": response.text,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }), 500


@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    """
    Analyze user-described symptoms using Gemini.

    Expects: JSON body with a 'symptoms' string field.
    Returns: JSON with analysis results.
    """
    try:
        # Check API key
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key not configured. Set the GEMINI_API_KEY environment variable."
            }), 500

        data = request.get_json()
        if not data or not data.get("symptoms", "").strip():
            return jsonify({"success": False, "error": "No symptoms provided."}), 400

        symptoms = data["symptoms"].strip()

        if len(symptoms) < 10:
            return jsonify({
                "success": False,
                "error": "Please provide a more detailed description (at least 10 characters)."
            }), 400

        if len(symptoms) > 5000:
            return jsonify({
                "success": False,
                "error": "Description too long. Please keep it under 5000 characters."
            }), 400

        # Build Gemini request
        model = get_gemini_model()

        prompt = (
            f"A user has described the following symptoms and health concerns:\n\n"
            f'"{symptoms}"\n\n'
            f"Please analyze these symptoms, discuss possible conditions, "
            f"provide health guidance, and advise when to seek professional medical help."
        )

        response = model.generate_content(prompt)

        return jsonify({
            "success": True,
            "analysis": response.text,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }), 500


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY environment variable is not set.")
        print("   Set it with: set GEMINI_API_KEY=your_api_key_here")
        print("   The app will start but AI analysis will not work.\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
