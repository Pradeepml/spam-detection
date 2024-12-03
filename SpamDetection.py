import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and preprocess data
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\mlporject1\mail_data.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['not spam', 'spam'])

mess = data['Message']
cat = data['Category']

# Split data
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

# Feature extraction
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Create and train model
model = MultinomialNB()
model.fit(features, cat_train)

# Test model accuracy
features_test = cv.transform(mess_test)

# Define prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result

# Streamlit UI
st.header('Spam Detection')

input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    if input_mess.strip():
        output = predict(input_mess)
        st.markdown(f"### Prediction: {output[0]}")
    else:
        st.markdown("### Please enter a message.")

# Optional: Display model accuracy
st.text(f"Model Accuracy: {model.score(features_test, cat_test):.2f}")
