import nltk
from nltk.corpus import movie_reviews
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# Download dataset
nltk.download('movie_reviews')
nltk.download('stopwords')  # Added stopwords download
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Check dataset distribution
labels = [category for _, category in documents]
print("🔹 Dataset Distribution:", Counter(labels))

# Text preprocessing (removes stopwords)
def clean_review(words):
    return " ".join([w.lower() for w in words if w.lower() not in stop_words])

text_data = [clean_review(words) for words, category in documents]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(text_data)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)  # Ensures balanced split

# Train the Model using Logistic Regression
model = LogisticRegression(max_iter=500)  # Increased iterations for better training
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Flask App for API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    print(f"\n🔹 Received Review: {data}")  # Debugging
    review_vector = vectorizer.transform([data])
    prediction = model.predict(review_vector)[0]
    print(f"🔹 Predicted Sentiment: {prediction}")  # Debugging
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Runs on port 8000

# Run the Flask API
# $ python app.py
# API will be accessible at http://
# Use tools like Postman or cURL to test the API
# POST Request: {"review": "The movie was fantastic!"}

# Sample cURL command
# curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"review\": \"This movie was fantastic!\"}"
# Output: {"sentiment":"pos"}
