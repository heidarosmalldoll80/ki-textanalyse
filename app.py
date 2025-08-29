from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Download the required VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get text from the submitted form
    text = request.form['text']
    # Analyze sentiment of the text
    sentiment = sentiment_analyzer.polarity_scores(text)
    # Vectorize the text to extract keywords
    vectorizer = CountVectorizer(stop_words='english')  # Changed to support English stop words
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.get_feature_names()
    # Return the sentiment and keywords as JSON
    return jsonify({'sentiment': sentiment, 'keywords': keywords.tolist()})

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)