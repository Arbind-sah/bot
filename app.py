import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopword removal
nltk.download('averaged_perceptron_tagger')  # For POS tagging

# Define chatbot functions
def clean_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return stemmed_words

def categorize_input(text):
    words = clean_text(text)
    pos_tags = pos_tag(words)
    
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    pronouns = [word for word, pos in pos_tags if pos in ['PRP', 'PRP$', 'WP', 'WP$']]
    verbs = [word for word, pos in pos_tags if pos.startswith('VB')]

    return nouns, pronouns, verbs

def respond(text):
    nouns, pronouns, verbs = categorize_input(text)
    
    if any(greeting in text.lower() for greeting in ["hi", "hello"]):
        return "Hello! How can I assist you today?"
    
    if "bye" in text.lower():
        return "Goodbye! Take care."

    if nouns:
        noun = nouns[0]
        return f"I don't know much about {noun}. Can you help me understand it?"
    
    if pronouns or verbs:
        topic = " ".join(verbs) or " ".join(pronouns)
        return f"I would like to know about {topic}."
    
    return "I'm not sure how to respond to that. Could you rephrase?"

# Start the chatbot
def start_chatbot():
    st.title("Text Mining Chatbot")
    user_input = st.text_input("Say something to the chatbot:")
    
    if user_input:
        response = respond(user_input)
        st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    start_chatbot()

