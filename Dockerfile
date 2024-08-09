FROM python:3.9

WORKDIR /app

# Create the models directory before copying
RUN mkdir -p /app/models

# Copy the Flask app code
COPY flask_app/ /app/

# Copy the vectorizer file to the models directory
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Install dependencies
COPY flask_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download NLTK stopwords and wordnet resources
RUN python -m nltk.downloader stopwords wordnet

# Expose the port the app runs on
EXPOSE 5000

# Use Gunicorn to serve the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
