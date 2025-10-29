import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib
from flask import Flask, request, render_template

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the function exactly as used when training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Flask app
app = Flask(__name__)

# Load pipeline
model = joblib.load('fake_news_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']
    result = model.predict([text])[0]
    message = "✅ This looks Real News" if result == 1 else "❌ This looks Fake News"
    return render_template('index.html', prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
