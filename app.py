from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = sentiment_analyzer.polarity_scores(text)
    vectorizer = CountVectorizer(stop_words='german')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return jsonify({'sentiment': sentiment, 'keywords': keywords.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
