import pickle
import logging
import os
from typing import Tuple, Any
from pathlib import Path

class SentimentModel:
    def __init__(self):
        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.classifier = None
        self._load_models()

    def _load_models(self) -> None:
        """Load all required ML models from pickle files."""
        try:
            # Get model path from environment or use default relative path
            default_path = str(Path(__file__).parent / 'Models')
            model_path = os.getenv('MODEL_PATH', default_path)
            logging.info(f"Loading models from: {model_path}")

            # Ensure the directory exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found at: {model_path}")

            model_files = {
                'count_vectorizer': 'count_vect.pkl',
                'tfidf_transformer': 'transformer.pkl',
                'classifier': 'Text_LR.pkl'
            }

            for model_name, filename in model_files.items():
                file_path = os.path.join(model_path, filename)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Model file not found: {file_path}")

                with open(file_path, 'rb') as f:
                    setattr(self, model_name, pickle.load(f))
                logging.info(f"Successfully loaded {model_name}")

        except FileNotFoundError as e:
            logging.error(f"Model file not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for given text using the ML pipeline.

        Args:
            text: Input text for sentiment analysis

        Returns:
            Tuple containing sentiment label and confidence score
        """
        try:
            if not all([self.count_vectorizer, self.tfidf_transformer, self.classifier]):
                raise RuntimeError("Models not properly initialized")

            # Transform text using CountVectorizer
            count_features = self.count_vectorizer.transform([text])
            logging.debug(f"Count features shape: {count_features.shape}")

            # Apply TF-IDF transformation
            tfidf_features = self.tfidf_transformer.transform(count_features)
            logging.debug(f"TF-IDF features shape: {tfidf_features.shape}")

            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(tfidf_features)[0]
            logging.debug(f"Raw prediction probabilities: {probabilities}")

            # Find the class with highest probability
            max_prob_idx = probabilities.argmax()
            confidence = probabilities[max_prob_idx]

            # Map the prediction index to sentiment
            # Class 2 (index 2) appears to be positive sentiment based on the logs
            sentiment = "positive" if max_prob_idx == 2 else "negative"

            logging.info(f"Prediction for text: '{text[:50]}...' -> {sentiment} (confidence: {confidence:.2f})")
            logging.debug(f"Probabilities - Positive: {confidence:.3f}")

            return sentiment, float(confidence)

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise