import streamlit as st
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from extractor import render_extractor_tab
from chatbot import render_chatbot_tab

# --- CONFIG ---
def get_api_key(key_name: str) -> str:
    """Get API key from secrets or environment variables"""
    try:
        return st.secrets[key_name]
    except:
        return os.getenv(key_name)

def initialize_apis():
    """Initialize OpenAI and Pinecone clients"""
    # Initialize OpenAI
    openai_api_key = get_api_key("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or as environment variable.")
        st.stop()
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Initialize Pinecone
    pinecone_api_key = get_api_key("PINECONE_API_KEY")
    pinecone_client = None
    pinecone_index = None
    
    if pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index_name = "whatsapp-qna"
            
            # Create index if it doesn't exist
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            pinecone_index = pc.Index(index_name)
            pinecone_client = pc
            
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
    else:
        st.warning("Pinecone API key not found. Knowledge base features will be limited.")
    
    return client, pinecone_client, pinecone_index

def main():
    # Set page config
    st.set_page_config(
        page_title="WhatsApp Knowledge System",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="💬"
    )
    
    # Global CSS styles
    st.markdown("""
        <style>
        /* Global styles */
        .main {
            padding: 1rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Typography */
        h1 {
            color: #ffffff !important;
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
            text-align: center;
        }
        
        .subtitle {
            color: #9ca3af;
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(30, 41, 59, 0.4);
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 1.1rem;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            border: 1px solid rgba(71, 85, 105, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.2) !important;
            color: #93c5fd !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Container styles */
        .content-box {
            background-color: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(71, 85, 105, 0.2);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Button styles */
        .stButton > button {
            background-color: rgba(59, 130, 246, 0.1) !important;
            color: #93c5fd !important;
            border: 1px solid rgba(59, 130, 246, 0.2) !important;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            background-color: rgba(59, 130, 246, 0.2) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #3b82f6 !important;
        }
        
        [data-testid="stMetricContainer"] {
            background-color: rgba(30, 41, 59, 0.5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        /* Alert/Info boxes */
        .stAlert {
            background-color: rgba(30, 41, 59, 0.5) !important;
            border: 1px solid rgba(100, 116, 139, 0.2) !important;
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            background-color: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.2);
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize APIs
    openai_client, pinecone_client, pinecone_index = initialize_apis()
    
    # Store in session state for access across tabs
    st.session_state.openai_client = openai_client
    st.session_state.pinecone_client = pinecone_client
    st.session_state.pinecone_index = pinecone_index
    
    # Header
    st.title("WhatsApp Knowledge Miner")
    st.markdown('<p class="subtitle">Extract knowledge from your WhatsApp messages and chat with your personal AI assistant</p>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📱 Message Extractor", "💬 Knowledge Chatbot"])
    
    with tab1:
        render_extractor_tab()
    
    with tab2:
        if pinecone_index:
            render_chatbot_tab()
        else:
            st.error("Knowledge base not available. Please check your Pinecone configuration.")
            st.info("The chatbot requires a working Pinecone connection to access stored knowledge.")

if __name__ == "__main__":
    main()