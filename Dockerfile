# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir flask flask-cors flask-sqlalchemy gunicorn numpy scikit-learn

# Copy the application code
COPY . /app/

# Create Models directory
RUN mkdir -p /app/Models

# Expose port 5000 for the Flask app
EXPOSE 7860

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--reload", "main:app"]
