from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)


# Set the secret key for development
app.secret_key = 'e591885bce2b1a2d19a6384c98c8fdf6d8a81c4df749e4a394b74f18e64cfa3b'


# Load the trained SVC model
svc_model = joblib.load("trained_model/svc_model.pkl")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
stemmer = PorterStemmer()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Load the training data
df = pd.read_csv('mental_health.csv')

# Preprocess the text data
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word.lower() not in set(stopwords.words('english'))]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Sentiment Analysis with VADER (Valence Aware Dictionary and sEntiment Reasoner)
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']  # Using compound score as a single sentiment score

df['sentiment_score'] = df['preprocessed_text'].apply(analyze_sentiment)

# Combine sentiment score with TF-IDF vectorization
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
X = tfidf_matrix
y = df['label']

# Train SVC model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc_model.fit(X_train, y_train)

# Route for the main page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')  # Use .get() to avoid KeyError
        password = request.form.get('password')

        if not username or not password:
            return "Missing username or password", 400  # Handle empty input gracefully

        # Set session and redirect (simplified example)
        session['user'] = username
        return redirect(url_for('index'))

    return render_template('login.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user already exists
        if username in users_db:
            return "User already exists", 400

        # Hash password and store user
        hashed_password = generate_password_hash(password, method='sha256')
        users_db[username] = {'password': hashed_password}
        return redirect(url_for('login'))  # Redirect to login

    return render_template('signup.html')

@app.route('/', methods=['GET'])
def home():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    return redirect(url_for('index'))  # Redirect to the index page if logged in


# Main page route (requires authentication)
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    prev_chat_message = "Hello There, <br> Share how you feel to predict your likely mental health state"
    return render_template('index.html', prev_chat_message=prev_chat_message)



# Route to handle form submission and display the predictive result
@app.route('/', methods=['POST'])
def predict():
    user_text = request.form['user_text']
    
    # Preprocess user input
    preprocessed_text = preprocess_text(user_text)
    
    # Calculate sentiment score
    sentiment_score = analyze_sentiment(preprocessed_text)
    
    # Transform preprocessed text into TF-IDF features
    tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    
    # Combine sentiment score with TF-IDF features
    combined_features = tfidf_text.copy()
    combined_features[0, -1] = sentiment_score
    
    # Make prediction using the loaded SVC model
    prediction = svc_model.predict(combined_features)[0]
    
    # Define the chat message based on the prediction
    if prediction == 1:
        prediction_result = "Oops!😟 You seem to be in a negative mental state."
    else:
        prediction_result = "Yas!😁 You seem to be in a positive mental state."
    
    # Retrieve previous chat message from the template
    prev_chat_message = request.form['prev_chat_message']
    
    return render_template('index.html', user_text=user_text, prev_chat_message=prev_chat_message, prediction_result=prediction_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=True)
