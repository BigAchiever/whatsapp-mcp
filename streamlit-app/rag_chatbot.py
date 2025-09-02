import streamlit as st
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import time

# --- CONFIG ---
def run_rag_chatbot(pinecone_index, client):
    def get_api_key(key_name: str) -> str:
        """Get API key from secrets or environment variables"""
        try:
            return st.secrets[key_name]
        except:
            return os.getenv(key_name)

    # Initialize OpenAI
    openai_api_key = get_api_key("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or as environment variable.")
        st.stop()
    client = openai.OpenAI(api_key=openai_api_key)

    # Initialize Pinecone
    pinecone_api_key = get_api_key("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("Pinecone API key not found.")
        st.stop()

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "whatsapp-qna"
    if index_name not in pc.list_indexes().names():
        st.error(f"Pinecone index '{index_name}' not found.")
        st.stop()

    index = pc.Index(index_name)

    # --- STREAMLIT UI ---
    st.set_page_config(
        page_title="WhatsApp Knowledge Chatbot", 
        layout="wide",
        page_icon="💬"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>

    .user-message {
        background-color: #1a1c24;
        padding: 12px 16px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .bot-message {
        background-color: #161d29;
        padding: 12px 16px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        margin-right: 20%;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #7f7f7f;
        text-align: right;
        margin-top: 5px;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.header('💬 WhatsApp Knowledge Chatbot')
    st.info('The chatbot refers to your whatapp knowledge base to answer.')

    # --- SESSION STATE ---
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'show_context' not in st.session_state:
        st.session_state.show_context = False

    # --- FUNCTIONS ---
    def get_query_embedding(query: str) -> List[float]:
        """Generate embedding for the user query"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding

    def retrieve_relevant_qna(query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant Q&A pairs from Pinecone"""
        query_vector = get_query_embedding(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        qna_list = []
        for match in results['matches']:
            qna_list.append(match['metadata'])
        return qna_list

    def generate_answer(query: str, retrieved_qna: List[Dict]) -> str:
        """Use ChatGPT to generate a natural language answer from retrieved Q&A"""
        if not retrieved_qna:
            context_text = "No relevant knowledge found in database."
        else:
            context_text = "\n".join([
                f"Group: {q['group_name']}\n"
                f"Date: {q.get('timestamp', 'N/A')}\n"
                f"Q: {q['prompt']}\nA: {q['completion']}\n"
                for q in retrieved_qna
            ])

        prompt = f"""
        You are an expert assistant with detailed knowledge of WhatsApp group discussions.
        A user has asked a question related to the 'Gate DA MATHS pro course' group.
        Use the following context to answer the question accurately. Do not fabricate any information. Provide clear, detailed explanations where applicable.

        Context:
        {context_text}

        User Question: {query}

        Answer naturally and informatively.

        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly, knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("About")
        st.info("This chatbot uses RAG (Retrieval Augmented Generation) to answer questions based on WhatsApp group discussions about the Gate DA MATHS pro course.")
        
        st.header("Controls")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
        st.session_state.show_context = st.checkbox("Show context sources", value=False)
        
        st.header("Information")
        st.write("The knowledge base contains curated Q&A pairs from WhatsApp group discussions.")

    # --- CHAT INTERFACE ---
    # Display chat history in a container
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f'<div class="user-message"><div class="message-header">You</div>{chat["user"]}</div>', unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f'<div class="bot-message"><div class="message-header">Assistant</div>{chat["bot"]}</div>', unsafe_allow_html=True)
                
                # Show context if enabled
                if st.session_state.show_context and chat.get("context"):
                    with st.expander(f"View sources for this response ({len(chat['context'])} found)"):
                        for j, context_item in enumerate(chat["context"]):
                            st.markdown(f"**Source {j+1}**")
                            st.markdown(f"**Group:** {context_item.get('group_name', 'N/A')}")
                            st.markdown(f"**Date:** {context_item.get('timestamp', 'N/A')}")
                            st.markdown(f"**Question:** {context_item.get('prompt', 'N/A')}")
                            st.markdown(f"**Answer:** {context_item.get('completion', 'N/A')}")
                            st.markdown("---")
                
            st.markdown('</div>', unsafe_allow_html=True)
    # Input area
    st.markdown("---")
    input_container = st.container()
    with input_container:
        col1, col2 = st.columns([6, 1], gap="small")
        with col1:
            if "user_input_temp" not in st.session_state:
                st.session_state.user_input_temp = ""
            user_input = st.text_input(
                "Ask your question here...", 
                st.session_state.user_input_temp,
                key="user_input_box",
                placeholder="Type your question about the course and press Enter or click Send"
            )
        with col2:
            send_button = st.button("Send", use_container_width=True)

    # Handle input
    if send_button and user_input.strip():
        with st.spinner("Thinking..."):
            # Save user input to session and clear temp input box
            st.session_state.user_input_temp = ""  # Clear text input after sending
            
            # Add user message to chat
            temp_user_msg = {"user": user_input, "bot": "Thinking...", "context": []}
            st.session_state.chat_history.append(temp_user_msg)
            
            # Retrieve relevant Q&A from Pinecone
            relevant_qna = retrieve_relevant_qna(user_input)
            
            # Generate answer
            answer = generate_answer(user_input, relevant_qna)
            
            # Update the last message
            st.session_state.chat_history[-1] = {
                "user": user_input,
                "bot": answer,
                "context": relevant_qna
            }
            
            # No need to rerun; chat history will update automatically
