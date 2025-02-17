import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import streamlit as st

# Download only necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

    if nouns:
        noun = nouns[0]
        return f"I don't know much about {noun}. Can you help me understand it?"
    
    if pronouns or verbs:
        topic = " ".join(verbs) or " ".join(pronouns)
        return f"I would like to know about {topic}."
    
    return "I'm not sure how to respond to that. Could you rephrase?"



def start_chatbot():
    st.set_page_config(page_title="Text Mining Chatbot", page_icon="🤖")
    st.title("🤖 Text Mining Chatbot")
    
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("### Say something to the chatbot:")
    user_input = st.text_input("", placeholder="Type your message here...")

    if 'questions' not in st.session_state:
        st.session_state['questions'] = []

    if user_input:
        st.session_state['questions'].append(user_input)
        response = respond(user_input)
        st.write(f"🤖 **Chatbot:** {response}")

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

    st.sidebar.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content h3 {
            color: #4CAF50;
        }
        .sidebar .sidebar-content p {
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    start_chatbot()