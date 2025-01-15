from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from dotenv import load_dotenv
import os

# Flask app initialization
app = Flask(__name__)

# Set a secret key for session management

# Load .env file
load_dotenv()

# Get the secret key
SECRET_KEY = os.getenv("SECRET_KEY")
# Set the secret key for the Flask app
app.secret_key = SECRET_KEY


# Load the trained models and vectorizers
svc_model_status = joblib.load("saved_models/tuned_svm_model_status.pkl")
svc_model_severity = joblib.load("saved_models/tuned_svm_model_severity.pkl")
tfidf_vectorizer_status = joblib.load("saved_models/tfidf_vectorizer_status.pkl")
tfidf_vectorizer_severity = joblib.load("saved_models/tfidf_vectorizer_severity.pkl")


# Load the LabelEncoders
status_encoder = joblib.load("saved_models/status_encoder.pkl")
severity_encoder = joblib.load("saved_models/severity_encoder.pkl")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
stemmer = PorterStemmer()

# Preprocessing function for user input
def preprocess_text(text):
    # Remove non-alphabetical characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)
    # Remove stopwords and stem words
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.lower() not in set(stopwords.words('english'))]
    return ' '.join(filtered_tokens)

# Route for the main page
# Sign-Up Page (Root)
@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get email and password from the form (but don't store or validate)
        email = request.form['email']
        password = request.form['password']

        # Proceed to the login page after signup
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Capture username from the form
        username = request.form.get('username')
        
        if not username:  # Ensure username is provided
            return "Username is required", 400
        
        # Set session with the username
        session['user'] = username

        # Redirect to the index page
        return redirect(url_for('index'))

    return render_template('login.html')  # Render login page on GET


# Index Page (Protected)
@app.route('/index')
def index():
    if 'user' not in session:  # Check if user is logged in
        return redirect(url_for('login'))  # Redirect to login if not
    prev_chat_message = "Hello There, <br> Share how you feel to predict your likely mental health state"
    return render_template('index.html', prev_chat_message=prev_chat_message)


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to categorize severity based on sentiment score from VADER
def get_severity_vader(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Mild"
    elif -0.2 <= score <= 0.2:
        return "Moderate"
    else:
        return "Severe"


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))  # Ensure the user is logged in

    user_text = request.form.get('user_text', '')

    if not user_text.strip():  # If user doesn't provide any input
        return render_template('index.html', chat_history=session.get('chat_history', []), 
                               error="Please enter some text to analyze.")
    
    # Preprocess the input text
    preprocessed_text = preprocess_text(user_text)

    # Vectorize the text for the status model
    tfidf_features_status = tfidf_vectorizer_status.transform([preprocessed_text])

    # Predict status using the SVM model
    prediction_status_encoded = svc_model_status.predict(tfidf_features_status)[0]

    # Decode the status prediction
    status_result = status_encoder.inverse_transform([prediction_status_encoded])[0]

    # Use VADER sentiment analysis to get an initial severity classification
    severity_result_vader = get_severity_vader(user_text)

    # Vectorize the text for the severity prediction model (SVM)
    tfidf_features_severity = tfidf_vectorizer_severity.transform([preprocessed_text])

    # Predict severity using the SVM model
    prediction_severity_encoded = svc_model_severity.predict(tfidf_features_severity)[0]

    # Decode the severity prediction from the SVM model
    severity_result_svm_encoded = severity_encoder.inverse_transform([prediction_severity_encoded])[0]

    # Combine severity from VADER and SVM
    if severity_result_vader != severity_result_svm_encoded:
        severity_result = severity_result_svm_encoded
    else:
        severity_result = severity_result_vader

    # Check for progression in status
    chat_history = session.get('chat_history', [])
    status_progression_message = ""

    if chat_history:
        last_status = chat_history[-1].get('status')
        if last_status and last_status != status_result:
            status_progression_message = f"Your mental status has changed from {last_status} to {status_result}."

    # Build a new chat entry
    new_chat_entry = {
        'user_text': user_text,
        'status': status_result,
        'severity': severity_result,
        'progression_message': status_progression_message
    }

    # Update the chat history in session
    chat_history.append(new_chat_entry)

    # Limit chat history to the last 10 entries
    chat_history = chat_history[-5:]

    # Save the updated chat history back to the session
    session['chat_history'] = chat_history

    # Render the updated chat interface with the conversation history
    return render_template('index.html', chat_history=chat_history)


# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
