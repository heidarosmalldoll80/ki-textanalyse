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
    """Analyze the sentiment of the provided text and extract keywords.

    Returns:
        JSON object containing sentiment scores and extracted keywords.
    """
    # Get text from the submitted form
    text = request.form['text']
    # Analyze sentiment of the text
    sentiment = sentiment_analyzer.polarity_scores(text)
    # Vectorize the text to extract keywords, ignoring English stop words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    # Get the feature names as keywords
    keywords = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.get_feature_names()  
    # Convert the keywords to a list to ensure proper format
    keywords_list = keywords.tolist() if hasattr(keywords, 'tolist') else list(keywords)
    # Return the sentiment and keywords as a JSON response
    return jsonify({'sentiment': sentiment, 'keywords': keywords_list})

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)