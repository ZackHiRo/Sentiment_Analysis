import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from models import SentimentModel
from utils import validate_input, setup_logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Setup logging
setup_logging()

# Initialize the sentiment model
sentiment_model = SentimentModel()

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler for all exceptions."""
    logging.error(f"Error occurred: {str(error)}")

    if isinstance(error, ValueError):
        return jsonify({"error": str(error)}), 400

    return jsonify({
        "error": "An internal error occurred. Please try again later."
    }), 500

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Endpoint for sentiment prediction.

    Expects JSON input with format:
    {
        "text": "text to analyze"
    }

    Returns:
    {
        "sentiment": "positive/negative",
        "confidence": float
    }
    """
    try:
        # Get and validate input
        data = request.get_json()
        if not data:
            raise ValueError("No input data provided")

        text = validate_input(data)

        # Get prediction
        sentiment, confidence = sentiment_model.predict(text)

        # Return response
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        }), 200

    except Exception as e:
        # Let the global error handler deal with it
        raise