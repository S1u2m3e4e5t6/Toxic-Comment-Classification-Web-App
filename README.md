# Toxic Message Detector 😠

This project is an NLP-based machine learning model designed to classify text messages or comments as either **Toxic** or **Not Toxic**. The model is built using Python and Scikit-learn and is trained on a dataset of Wikipedia comments.

---

## 🚀 Features

* Processes and cleans raw text data for machine learning.
* Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.
* Trained with a **Logistic Regression** classifier, a reliable and efficient model for text classification.
* Achieves approximately **95% accuracy** on the test set.

---

## 🛠️ Technology Stack

* **Python**: The core programming language.
* **Pandas**: For data loading and manipulation.
* **NLTK (Natural Language Toolkit)**: For text cleaning, stop word removal, and lemmatization.
* **Scikit-learn**: For machine learning, including TF-IDF, model training, and evaluation.
* **Google Colab**: As the development environment to bypass local permission issues and leverage cloud computing power.

---

## ⚙️ Setup and Usage

This project is designed to be run easily in Google Colab, which requires no local setup.

1.  **Go to Google Colab**
    Open [https://colab.research.google.com/](https://colab.research.google.com/) and select **File -> New notebook**.

2.  **Upload the Dataset**
    * Download the `train.csv` file from the [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).
    * In your Colab notebook, click the **folder icon** on the left, then click the **upload button** and select your `train.csv` file.

3.  **Run the Code**
    Copy the complete Python script into a cell in your notebook and run it. The script will handle everything from data loading and cleaning to model training and evaluation.

---

## 📊 Model Performance

The model was trained on 100,000 comment samples and achieved the following results on the test set:

* **Overall Accuracy**: `95.30%`

### Classification Report

```
              precision    recall  f1-score   support

   Not Toxic       0.96      0.99      0.97     17983
       Toxic       0.92      0.58      0.72      2017

    accuracy                           0.95     20000
   macro avg       0.94      0.79      0.84     20000
weighted avg       0.95      0.95      0.95     20000
```

**Analysis**: The model is highly precise, meaning its positive predictions are usually correct. However, the lower recall for the "Toxic" class (0.58) indicates it sometimes fails to identify a toxic comment. This is a common challenge with imbalanced datasets and a key area for future improvement.

---

## 🔮 Making Predictions

You can use the trained model to classify new sentences. The following function demonstrates how:

```python
# Function to predict toxicity of a new comment
def predict_toxicity(comment):
    # The function cleans the text, converts it to a TF-IDF vector,
    # and uses the trained model to make a prediction.
    cleaned_comment = clean_text(comment)
    comment_vector = vectorizer.transform([cleaned_comment])
    prediction = model.predict(comment_vector)

    if prediction[0] == 1:
        print(f"\n'{comment}' -> Prediction: TOXIC 😠")
    else:
        print(f"\n'{comment}' -> Prediction: Not Toxic 😊")

# --- Example Usage ---
predict_toxicity("you are a wonderful person and I love your content")
predict_toxicity("go away you are the worst person ever")
predict_toxicity("I will find you and hurt you")
```

### Example Output

When you run the code above, you will get the following output, which clearly shows the model's limitations:

```
'you are a wonderful person and I love your content' -> Prediction: Not Toxic 😊

'go away you are the worst person ever' -> Prediction:  Toxic 😠

'I will find you and hurt you' -> Prediction: Toxic 😠
```
**Note**: Notice that the model incorrectly classifies some clearly toxic comments. This is a direct result of the model's low recall (0.58) for the toxic class, as shown in the performance report. Improving this is the main goal for future versions of the model.
