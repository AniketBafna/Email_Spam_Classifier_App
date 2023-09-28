import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
stopwords.words('english')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    
    ## To lower the text
    text = text.lower()
    
    ## to break the sentences into words
    text = nltk.word_tokenize(text)
    
    ## If the words is Alpha-Numeric the append
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text= y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)



def model():
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))

    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_input("Enter the message")

    if st.button('Predict'):

        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        if result ==1:
            st.header("Spam")
        else:
            st.header("Not Spam")
