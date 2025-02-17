import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import streamlit as st
import re
import joblib

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    return lemmatized_words




def categorize_input(text):
    words = clean_text(text)
    pos_tags = pos_tag(words)
    
    categorized_words = {
        'nouns': [word for word, pos in pos_tags if pos.startswith('NN')],
        'pronouns': [word for word, pos in pos_tags if pos in ['PRP', 'PRP$', 'WP', 'WP$']],
        'verbs': [word for word, pos in pos_tags if pos.startswith('VB')]
    }
    nouns = categorized_words['nouns']
    pronouns = categorized_words['pronouns']
    verbs = categorized_words['verbs']

    return nouns, pronouns, verbs




def respond(text):
    nouns, pronouns, verbs = categorize_input(text)
    
    greetings = ["hi", "hello", "hey", "hola", "greetings", "what's up", "sup", "yo"]
    if any(greeting in text.lower() for greeting in greetings):
        return "Hello! How can I assist you today?"
    
    farewells = ["bye", "goodbye", "see you", "take care", "later", "farewell"]
    if any(farewell in text.lower() for farewell in farewells):
        return "Goodbye! Take care."

    age = extract_age(text)
    if age:
        return predict_salary(text)

    if nouns:
        noun = nouns[0]
        return f"I don't know much about {noun}. Can you help me understand it?"
    
    if pronouns or verbs:
        topic = " ".join(verbs) or " ".join(pronouns)
        return f"I would like to know about {topic}."
    return "I'm not sure how to respond to that. Could you rephrase?"

def extract_age(text):
    pattern = r'(\d{1,3})\s*(?:years? old|years|yo|age|aged|yr|yrs|y/o)?'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    return None

# Load model
model = joblib.load("salary_prediction_model.pkl")



def predict_salary(text):
    age = extract_age(text)
    if age:
        salary = model.predict([[age]])[0]
        return f"Predicted salary for age {age}: ${salary:.2f}"
    return "No age detected in input."



def start_chatbot():
    st.set_page_config(page_title="Text Mining Chatbot", page_icon="🤖")
    st.title("🤖 Text Mining Chatbot")

    st.write("### Say something to the chatbot:")
    user_input = st.text_input("", placeholder="Type your message here...")

    if 'questions' not in st.session_state:
        st.session_state.questions = []

    if user_input:
        st.session_state['questions'].append(user_input)
        response = respond(user_input)
        st.write(f"🤖: {response}")

    st.sidebar.title("User Questions")
    for question in st.session_state['questions']:
        st.sidebar.write(question)
    
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Home", "Settings", "About"])

    if tab == "Home":
        st.sidebar.markdown("### Welcome to the Home tab!")
        st.sidebar.markdown("This is the main area where you can interact with the chatbot.")

    elif tab == "Settings":
        st.sidebar.markdown("### Settings")
        st.sidebar.markdown("Here you can adjust the settings of the chatbot.")

    elif tab == "About":
        st.sidebar.markdown("### About")
        st.sidebar.markdown("This chatbot is designed to help you with text mining tasks.")

if __name__ == "__main__":
    start_chatbot()
