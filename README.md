# Email-Spam-Or-Ham-Classification-By-NLP

## Aim

The aim of this project is to develop an efficient and accurate machine learning model to classify email messages as either spam or ham (non-spam).

This project demonstrates the process of generating, preprocessing, vectorizing, and classifying email text data to predict whether an email is spam or not using a Multinomial Naive Bayes classifier.

## Overview

This project involves several key steps:

1. **Text Generation**: Construct email text based on word frequencies.
2. **Preprocessing**: Clean and preprocess the text data.
3. **Vectorization**: Convert text data into numerical features using CountVectorizer.
4. **Model Training**: Train a Multinomial Naive Bayes classifier.
5. **Evaluation**: Evaluate the model using accuracy, confusion matrix, and classification report.
6. **Prediction**: Predict whether a new email is spam or ham.


### Main Steps 

1. **Data Preparation**:
   - Create a DataFrame with email word frequencies.
   - Generate email texts based on word frequencies.

2. **Preprocessing**:
   - Remove punctuation and numbers.
   - Convert text to lowercase.
   - Tokenize the text and remove stopwords.

3. **Vectorization**:
   - Transform the preprocessed text data into numerical feature vectors.

4. **Model Training and Evaluation**:
   - Train a Multinomial Naive Bayes classifier.
   - Evaluate the model using accuracy, confusion matrix, and classification report.

5. **Prediction**:
   - Predict if a new email is spam or ham based on raw text.

## Dataset

The dataset used in this project is a simulated dataset with word frequencies for different emails. The dataset contains the following columns:
- `Email No.`: Identifier for each email.
- Word columns: Frequency of each word in the email.
- `Prediction`: Binary label indicating if the email is spam (1) or not (0).

## Model Training and Evaluation

- **Train-Test Split**: Split the data into training and testing sets.
- **Model**: Train a Multinomial Naive Bayes classifier.
- **Evaluation**: Evaluate the model on the test set using accuracy, confusion matrix, and classification report.

## Prediction

The `predict_email` function takes a new email text, preprocesses it, transforms it using the trained vectorizer, and predicts if it's spam or ham using the trained model.

def predict_email(model, vectorizer, email_text):
    # Preprocess the email text
    processed_text = preprocess_text(email_text)
    # Transform the text using the trained vectorizer
    vectorized_text = vectorizer.transform([processed_text])
    # Make a prediction using the trained model
    prediction = model.predict(vectorized_text)
    # Map the prediction to a label
    return 'spam' if prediction[0] == 1 else 'ham'

