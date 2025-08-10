import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk

# --- Step 0: Initial Setup ---
print("Setting up NLTK...")
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK setup complete.")


# --- Step 1: Load a LARGER dataset ---
print("\nLoading data... (using a larger sample of 100,000 rows)")
# We increased nrows to 100,000 to build a better model.
# Using the explicit path to train.csv
df = pd.read_csv('/train.csv', nrows=100000)
print("Data loaded successfully.")

# Create a single 'toxic' label
toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['toxic'] = (df[toxic_cols].sum(axis=1) > 0).astype(int)

# Define the text cleaning function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply the cleaning function
print("\nCleaning and preprocessing text...")
df['cleaned_text'] = df['comment_text'].apply(clean_text)
print("Text cleaning complete.")


# --- Step 2: Feature Extraction (TF-IDF) ---
print("\nConverting text to numbers using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['toxic']
print("Text vectorization complete.")


# --- Step 3: Train the Machine Learning Model ---
print("\nSplitting data and training the model...")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")


# --- Step 4: Evaluate the Model ---
print("\nEvaluating the model performance...")
# Make predictions on the test data
y_pred = model.predict(X_test)

# Print the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Not Toxic', 'Toxic'])

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
