from typing import Dict, Any
import logging

def validate_input(data: Dict[str, Any]) -> str:
    """
    Validate the input data for sentiment analysis.
    
    Args:
        data: Dictionary containing the input data
        
    Returns:
        Validated text string
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
        
    if 'text' not in data:
        raise ValueError("Missing 'text' field in input")
        
    text = data.get('text')
    
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
        
    if not text.strip():
        raise ValueError("Text cannot be empty")
        
    return text.strip()

def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
