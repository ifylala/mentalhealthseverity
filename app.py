from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Flask app initialization
app = Flask(__name__)

# Set a secret key for session management
app.secret_key = 'e591885bce2b1a2d19a6384c98c8fdf6d8a81c4df749e4a394b74f18e64cfa3b'

# Load the trained models and vectorizers
svc_model_status = joblib.load("saved_models/svm_model_status.pkl")
svc_model_severity = joblib.load("saved_models/svm_model_severity.pkl")
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


# Route to handle form submission and display the predictive result
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_text = request.form.get('user_text', '')
    if not user_text.strip():
        return render_template('index.html', prev_chat_message="Please enter some text to analyze.")

    # Preprocess user input
    preprocessed_text = preprocess_text(user_text)

    # Vectorize text for both models
    tfidf_features_status = tfidf_vectorizer_status.transform([preprocessed_text])
    tfidf_features_severity = tfidf_vectorizer_severity.transform([preprocessed_text])

    # Predict using the SVM models
    prediction_status_encoded = svc_model_status.predict(tfidf_features_status)[0]
    prediction_severity_encoded = svc_model_severity.predict(tfidf_features_severity)[0]

    # Decode the predictions to the actual labels
    status_result = status_encoder.inverse_transform([prediction_status_encoded])[0]
    severity_result = severity_encoder.inverse_transform([prediction_severity_encoded])[0]

    # Map severity levels (optional, if necessary)
    if severity_result == 0:
        severity_result = "Mild"
    elif severity_result == 1:
        severity_result = "Moderate"
    elif severity_result == 2:
        severity_result = "Severe"

    # Return the result to the user
    prev_chat_message = f"You said: {user_text}<br>Status: {status_result}<br>Severity: {severity_result}"
    return render_template('index.html', prev_chat_message=prev_chat_message)




# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
