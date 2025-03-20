from app import app
import logging
from utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting sentiment analysis API server")
    app.run(host="0.0.0.0", port=5000, debug=True)
