import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import cloudpickle
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
with open('fake_news_vectorizer.pkl', 'rb') as f:
    vectorizer = cloudpickle.load(f)
with open('fake_news_model.pkl', 'rb') as f:
    model = cloudpickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    result = model.predict(vect)[0]
    message = "✅ This looks Real News" if result == 1 else "❌ This looks Fake News"
    return render_template('index.html', prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
