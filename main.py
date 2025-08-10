import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk

print("Setting up NLTK...")
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK setup complete.")


print("\nLoading data... (using a larger sample of 100,000 rows)")

df = pd.read_csv('/train.csv', nrows=100000)
print("Data loaded successfully.")

toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['toxic'] = (df[toxic_cols].sum(axis=1) > 0).astype(int)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("\nCleaning and preprocessing text...")
df['cleaned_text'] = df['comment_text'].apply(clean_text)
print("Text cleaning complete.")


print("\nConverting text to numbers using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['toxic']
print("Text vectorization complete.")


print("\nSplitting data and training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")


print("\nEvaluating the model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Not Toxic', 'Toxic'])

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)



def predict_toxicity(comment):
    cleaned_comment = clean_text(comment)
    
    comment_vector = vectorizer.transform([cleaned_comment])
    
    prediction = model.predict(comment_vector)
    
    if prediction[0] == 1:
        print(f"\n'{comment}' -> Prediction: TOXIC 😠")
    else:
        print(f"\n'{comment}' -> Prediction: Not Toxic 😊")

predict_toxicity("you are a wonderful person and I love your content")
predict_toxicity("go away you are the worst person ever")
predict_toxicity("I will find you and hurt you")

# Try your own!
# replace (you are a wonderful person and I love your content)=(your own).
# thanks for your time have a wounderfull day!
