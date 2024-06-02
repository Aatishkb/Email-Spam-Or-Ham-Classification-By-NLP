# **Email Spam Or Ham Classification By NLP**
"""

i = Image.open(r'C:\Users\kumar\Desktop\email.jfif') i

"""# **(1). Importing Required Libraries**"""

import pandas as pd
from PIL import Image
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

"""# **(2) . Load the dataset**"""

df = pd.read_csv(r'A:\MTECH(Data Science)\DataSet\P\emails.csv')
df.sample(5)

df.shape

df.info()

"""# **(3) . Fetch Columns Excluding 'Email No.' and 'Prediction'**"""

word_columns = df.columns[1:-1]
word_columns

"""# **(4) . Generate text for each email**"""

texts = []
for index, row in df.iterrows():
    words = []
    for word in word_columns:
        frequency = row[word]
        if frequency > 0:
            words.extend([word] * frequency)
    email_text = ' '.join(words)
    texts.append(email_text)

texts

"""# **(5). Print the generated text for each email**"""

# for email_no, text in enumerate(texts):
    # print(f"{email_no}: {text}\n")
# Print only the first 10 entries to avoid exceeding the data rate limit
for email_no, text in enumerate(texts[:10]):
    print(f"{email_no}: {text}\n")

"""# **(6). Create a new DataFrame with 'Email No.', 'text', and 'Prediction' columns**"""

new_df = pd.DataFrame({
    'Email No.': df['Email No.'],
    'text': texts,
    'Prediction': df['Prediction']
})

new_df

"""# **(7). Function to preprocess the text**
  - i. Remove punctuation and numbers
  - ii. Convert to lowercase
  - iii. Tokenize the text
  - iv. Remove stopwords
"""

def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Join the words back into a single string
    return ' '.join(words)

"""# **(8) . Apply preprocessing to the text column**"""

new_df['text'] = new_df['text'].apply(preprocess_text)

"""# **(9) . Create a CountVectorizer object**"""

vectorizer = CountVectorizer()

"""# **(10) . Transform the text data into feature vectors**"""

X = vectorizer.fit_transform(new_df['text'])

X

"""# **(11) . Convert 'Prediction' to binary format**"""

y = new_df['Prediction']

y

"""# **(12) . Split the data into training and testing sets**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train

y_train

"""# **(13) . Create a Multinomial Naive Bayes classifier**"""

model = MultinomialNB()

"""# **(14) . Train the classifier**"""

model.fit(X_train, y_train)

"""# **(15) . Make predictions on the test set**"""

y_pred = model.predict(X_test)

"""# **(16) . Calculate accuracy**"""

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

"""# **(17) . Print confusion matrix and classification report**"""

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

"""# **(18) . Function to predict if a new email is spam or ham based on raw text**"""

def predict_email(model, vectorizer, email_text):
    # Preprocess the email text
    processed_text = preprocess_text(email_text)
    # Transform the text using the trained vectorizer
    vectorized_text = vectorizer.transform([processed_text])
    # Make a prediction using the trained model
    prediction = model.predict(vectorized_text)
    # Map the prediction to a label
    return 'spam' if prediction[0] == 1 else 'ham'

"""# **(19) . predict if a new email is spam or ham**"""

new_email = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
result = predict_email(model, vectorizer, new_email)
print(f'The email is classified as: {result}')

new_email = """Subject: Important Update on Your Account

Dear [Recipient's Name],

We hope this email finds you well.

We are writing to inform you about some important updates to your account. Recently, we have made improvements to our security systems to better protect your personal information. As part of this enhancement, we recommend that you review your account settings and update your password to ensure it meets our new security standards.

To update your password, please follow these steps:
1. Log in to your account on our website.
2. Navigate to the "Account Settings" section.
3. Click on "Change Password" and follow the on-screen instructions.

If you have any questions or need assistance, our customer support team is available 24/7. You can reach us by replying to this email or calling our support hotline at 1-800-123-4567.

Thank you for your continued trust in our services.

Best regards,
[Your Company's Name]
Customer Support Team
"""

result = predict_email(model, vectorizer, new_email)
print(f'The email is classified as: {result}')

new_email = """Subject: URGENT: You've Won a $1,000 Walmart Gift Card!

Congratulations [Recipient's Name]!

You are the lucky winner of a $1,000 Walmart Gift Card! To claim your prize, all you have to do is click the link below and provide your information.

>> Click here to claim your $1,000 Walmart Gift Card now! <<

Hurry! This offer is only valid for the next 24 hours. Don't miss out on this amazing opportunity!

Best regards,
The Walmart Rewards Team
"""

result = predict_email(model, vectorizer, new_email)
print(f'The email is classified as: {result}')

"""# **************************************************************************

## Name - Aatish Kumar Baitha
  - M.Tech(Data Science 2nd Year Student)
- My Linkedin Profile -
  - https://www.linkedin.com/in/aatish-kumar-baitha-ba9523191
- My Blog
  - https://computersciencedatascience.blogspot.com/
- My Github Profile
  - https://github.com/Aatishkb

# **Thank you!**

# **************************************************************************
"""
