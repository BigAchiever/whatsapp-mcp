import streamlit as st
import sqlite3
import json
import pandas as pd
import os
import openai
from datetime import datetime
from typing import List, Dict, Any
import time
from rag_chatbot import run_rag_chatbot

# --- CONFIG ---
# Initialize API keys
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

def generate_qna_pairs(message: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate Q&A pairs from a message using GPT-4"""
    try:
        context = f"""
        Message from: {"You" if message['is_from_me'] else message['sender']}
        Time: {message['timestamp']}
        Content: {message['message']}
        Media: {"Yes - " + message['media_type'] if message['media_type'] else "No"}
        """
        
        prompt = f"""Analyze this WhatsApp message for knowledge-worthy content:
        {context}
        
        Instructions:
        1. Extract ALL possible factual Q&A pairs that can be derived from the message.
        2. Do NOT invent or hallucinate information. Only use what is explicitly present.
        3. Do NOT repeat the same Q&A in different words.
        4. Skip trivial, casual, or meaningless content (like greetings, emojis, small talk).
        5. Each Q&A should be specific, detailed, and knowledge-worthy (dates, numbers, names, facts).
        6. Output as JSON objects, one per line (JSONL format).
        
        If the message isn't knowledge-worthy (like casual greetings, acknowledgments, or routine messages), return an empty list.
        
        Format each Q&A as JSON objects like:
        {{"prompt": "Q: [question]\\n\\n", "completion": "A: [answer]\\n"}}
        Return only the JSON objects, one per line (JSONL format)."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a knowledge extraction expert. 
                 Your task is to identify valuable information from chat messages and create precise, contextual Q&A pairs."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        if not content:
            return []
        
        qna_pairs = []
        for line in content.splitlines():
            try:
                qna = json.loads(line.strip())
                if "prompt" in qna and "completion" in qna:
                    qna_pairs.append(qna)
            except json.JSONDecodeError:
                continue

        return qna_pairs

    except Exception as e:
        st.error(f"Error generating Q&A pairs: {str(e)}")
        return []

def qna_pairs_to_jsonl(qna_pairs: list[dict]) -> str:
    """Convert list of Q&A dicts to JSONL string"""
    lines = [json.dumps(qna, ensure_ascii=False) for qna in qna_pairs]
    return "\n".join(lines)

DB_PATH = "../whatsapp-bridge/store/messages.db"  

# Set page config and styling
st.set_page_config(
    page_title="WhatsApp Message Extractor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    /* Global styles */
    .main {
        padding: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    h2 {
        color: #e5e7eb !important;
        font-size: 1.8rem !important;
        font-weight: 500 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    h3 {
        color: #d1d5db !important;
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
    }
    .description {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Container styles */
    .content-box {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Message styles */
    .message-container {
        background-color: rgba(15, 23, 42, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid rgba(71, 85, 105, 0.2);
    }
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(71, 85, 105, 0.2);
    }
    .sender {
        color: #e2e8f0;
        font-weight: 500;
    }
    .timestamp {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    .message-content {
        color: #cbd5e1;
        line-height: 1.5;
    }
    .media-info {
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-top: 0.75rem;
        color: #93c5fd;
        font-size: 0.875rem;
    }
    
    /* Action buttons */
    .stButton > button {
        width: 100%;
        background-color: rgba(59, 130, 246, 0.1) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    .stButton > button:hover {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Stats */
    .stats-box {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        color: #60a5fa;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton > button {
        width: 100%;
    }
    .stProgress > div > div > div {
        background-color: #3b82f6;
    }
    /* Stats cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #3b82f6 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    [data-testid="stMetricContainer"] {
        background-color: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    /* Message containers */
    div[data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(100, 116, 139, 0.2);
        margin-bottom: 0.5rem;
    }
    /* Container styling */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
        padding: 0.5rem;
    }
    /* Info boxes */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(100, 116, 139, 0.2) !important;
    }
    /* Dividers */
    .stDivider {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Group details container */
    div[data-testid="stNotificationContentSuccess"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(100, 116, 139, 0.2);
        color: #e2e8f0;
    }
    /* Code blocks */
    div[data-testid="stCodeBlock"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    </style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---
def get_groups():
    """Fetch distinct group JIDs with names from DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT DISTINCT m.chat_jid, c.name
            FROM messages m
            LEFT JOIN chats c ON m.chat_jid = c.jid
            WHERE m.chat_jid LIKE '%@g.us'
        """)
        
        groups = []
        for row in cursor.fetchall():
            chat_jid = row[0]
            group_name = row[1] if row[1] else f"Group {chat_jid[-8:]}"
            
            groups.append({
                "jid": chat_jid,
                "name": group_name
            })
        
        groups.sort(key=lambda x: x["name"])
        
    except Exception as e:
        st.warning(f"Could not fetch group names: {e}. Using JIDs instead.")
        cursor.execute("SELECT DISTINCT chat_jid FROM messages WHERE chat_jid LIKE '%@g.us'")
        groups = [{"jid": row[0], "name": f"Group {row[0][-8:]}"} for row in cursor.fetchall()]
    
    conn.close()
    return groups

def get_group_messages(group_jid, limit=100):
    """Fetch messages from a specific group"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sender, content, timestamp, is_from_me, media_type, filename
        FROM messages
        WHERE chat_jid = ?
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (group_jid, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "timestamp": r[2], "is_from_me": r[3], "media_type": r[4], "filename": r[5]} for r in rows]

# --- STREAMLIT UI ---
st.title("WhatsApp Message Extractor")
st.markdown('<p class="description">Extract, analyze, and generate knowledge from your WhatsApp group messages using AI-powered processing.</p>', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = None
if 'current_group' not in st.session_state:
    st.session_state.current_group = None
if 'qna_pairs' not in st.session_state:
    st.session_state.qna_pairs = []

# Initialize Pinecone
pinecone_api_key = get_api_key("PINECONE_API_KEY")
if pinecone_api_key:
    if 'pinecone' not in st.session_state:
        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            index_name = "whatsapp-qna"
            
            # Create index if it doesn't exist
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            st.session_state.pinecone = pc.Index(index_name)
            st.success("Successfully connected to Pinecone knowledge base!")
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.session_state.pinecone = None
else:
    st.warning("Pinecone API key not found. Knowledge base storage will be disabled.")

# Main content area
groups = get_groups()

if not groups:
    st.warning("⚠️ No groups found in database.")
    st.info("Make sure your WhatsApp database is properly connected and contains group messages.")
else:
    # Group selection
    st.subheader("📱 Select WhatsApp Group")
    
    selected_group = st.selectbox(
        "Choose a group to analyze:",
        options=groups,
        format_func=lambda g: f"{g['name']} • {g['jid'][-12:]}",
        help="Select the WhatsApp group you want to extract messages from"
    )
    
    if selected_group:
        # Group info card
        with st.container():
            st.markdown("**Selected Group Details**")
            st.markdown(f"**Name:** {selected_group['name']}")
            st.markdown(f"**ID:** {selected_group['jid']}")
        
        # Fetch messages button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Load Messages", type="primary", use_container_width=True):
                with st.spinner("Fetching messages..."):
                    st.session_state.messages = get_group_messages(selected_group["jid"], limit=200)
                    st.session_state.current_group = selected_group["jid"]
                    st.session_state.qna_pairs = []  # Reset Q&A pairs when loading new messages
                    st.rerun()
        
        # Display messages if loaded
        if st.session_state.messages and st.session_state.current_group == selected_group["jid"]:
            messages = st.session_state.messages
            
            # Statistics
            st.markdown("---")
            st.subheader("Statistics")
            
            # Calculate stats
            total_messages = len(messages)
            media_messages = len([m for m in messages if m.get('media_type')])
            unique_senders = len(set([m['sender'] for m in messages if not m['is_from_me']]))
            
            # Display stats in a more visual way
            stats_container = st.container()
            with stats_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Messages",
                        f"{total_messages:,}",
                        help="Total number of messages in this group"
                    )
                with col2:
                    st.metric(
                        "Media Messages",
                        f"{media_messages:,}",
                        delta=f"{(media_messages/total_messages)*100:.1f}%" if total_messages > 0 else None,
                        help="Number of messages containing media (images, videos, etc.)"
                    )
                with col3:
                    st.metric(
                        "Unique Senders",
                        f"{unique_senders:,}",
                        help="Number of different people who sent messages"
                    )
                with col4:
                    st.metric(
                        "Q&A Pairs",
                        f"{len(st.session_state.qna_pairs):,}",
                        help="Number of knowledge Q&A pairs generated"
                    )
            
            # Messages display
            st.markdown("---")
            st.subheader("💬 Recent Messages")
            
            # Show first 20 messages by default
            show_all = st.checkbox("Show all messages", value=False)
            display_messages = messages if show_all else messages[:20]
            
            for msg in display_messages:
                sender_display = "You" if msg['is_from_me'] else (msg['sender'] or "Unknown")
                
                with st.expander(f"Message from {sender_display} • {msg['timestamp']}", expanded=False):
                    st.markdown(msg['message'] if msg['message'] else "*No text content*")
                    
                    if msg.get('media_type'):
                        with st.container():
                            st.info(
                                f"**Media Type:** {msg['media_type']}" + 
                                (f"\n\n**File:** {msg['filename']}" if msg.get('filename') else ""),
                                icon="📎"
                            )
            
            if not show_all and len(messages) > 10:
                st.info(f"Showing first 10 messages. Check 'Show all messages' to see all {len(messages)} messages.")
            
            # Export section
            st.markdown("---")
            
            # AI Knowledge Generation section
            with st.container():
                st.header("AI Knowledge Generation")
                st.markdown('<p class="description">Transform your messages into a structured knowledge base using AI analysis.</p>', unsafe_allow_html=True)
                
                with st.container():
                    if st.button("🔍 Generate Knowledge Q&A", type="primary", use_container_width=True):
                        with st.spinner("Analyzing messages with AI..."):
                            all_qna_pairs = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, message in enumerate(messages):
                                if message.get('message'):
                                    status_text.text(f"Processing message {i+1}/{len(messages)}...")
                                    qna_pairs = generate_qna_pairs(message)
                                    if qna_pairs:
                                        all_qna_pairs.extend(qna_pairs)
                                progress_bar.progress((i + 1) / len(messages))
                            
                            st.session_state.qna_pairs = all_qna_pairs
                            progress_bar.empty()
                            status_text.empty()
                            
                            if all_qna_pairs:
                                st.success(f"Successfully generated {len(all_qna_pairs)} knowledge Q&A pairs!")
                            else:
                                st.info("No knowledge-worthy content found in the analyzed messages")
                            st.rerun()
                # Display Q&A preview
                with st.expander(f"View {len(st.session_state.qna_pairs)} Q&A Pairs", expanded=False):
                    for i, qna in enumerate(st.session_state.qna_pairs[:10], 1):  # Show first 10
                        st.markdown(f"**Q&A Pair {i}**")
                        st.code(f"{qna['prompt']}{qna['completion']}", language="text")
                        if i < len(st.session_state.qna_pairs[:10]):
                            st.divider()
                    
                    if len(st.session_state.qna_pairs) > 10:
                        st.info(f"Showing first 10 Q&A pairs. Download the full set to see all {len(st.session_state.qna_pairs)} pairs.")
                # download jsonl
                if st.session_state.qna_pairs:
                    jsonl_content = qna_pairs_to_jsonl(st.session_state.qna_pairs)
                    
                    st.download_button(
                        label="Download Q&A JSONL",
                        data=jsonl_content,
                        file_name=f"whatsapp_qna_{selected_group['jid']}.jsonl",
                        mime="application/json",
                        use_container_width=True
                    )

                # Preview Q&A pairs if they exist
                if st.session_state.qna_pairs:
                    st.markdown("---")
                    st.header("Store Q&A in Database")
                    
                    # # Add knowledge base search
                    # cols = st.columns([2, 1])
                    # with cols[0]:
                    #     search_query = st.text_input("🔍 Search stored knowledge", placeholder="Enter your question...")
                    # with cols[1]:
                    #     top_k = st.number_input("Number of results", min_value=1, max_value=10, value=3)
                    
                    # if search_query and st.session_state.pinecone:
                    #     with st.spinner("Searching knowledge base..."):
                    #         try:
                    #             # Generate embedding for search query
                    #             openai_client = openai.OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
                    #             response = openai_client.embeddings.create(
                    #                 model="text-embedding-ada-002",
                    #                 input=search_query
                    #             )
                    #             query_embedding = response.data[0].embedding
                                
                    #             # Search Pinecone
                    #             results = st.session_state.pinecone.query(
                    #                 vector=query_embedding,
                    #                 top_k=top_k,
                    #                 include_metadata=True
                    #             )
                                
                    #             if results.matches:
                    #                 st.markdown("#### Search Results")
                    #                 for i, match in enumerate(results.matches, 1):
                    #                     metadata = match.metadata
                    #                     with st.expander(f"Result {i} - Score: {match.score:.2f}", expanded=True):
                    #                         st.markdown(f"**From group:** {metadata['group_name']}")
                    #                         st.code(f"{metadata['prompt']}{metadata['completion']}", language="text")
                    #             else:
                    #                 st.info("No matching knowledge found in the database.")
                                    
                    #         except Exception as e:
                    #             st.error(f"Error searching knowledge base: {str(e)}")


                    # Add Pinecone storage button
                    if st.session_state.pinecone:
                        if st.button("Store Q&A Pairs in Knowledge Base", type="secondary", use_container_width=True):
                            with st.spinner("Storing Q&A pairs in knowledge base..."):
                                try:
                                    openai_client = openai.OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
                                    
                                    # Generate embeddings and store in batches
                                    batch_size = 100
                                    total_stored = 0
                                    
                                    for i in range(0, len(st.session_state.qna_pairs), batch_size):
                                        batch = st.session_state.qna_pairs[i:i + batch_size]
                                        batch_texts = []
                                        
                                        # Prepare texts for embedding
                                        for qna in batch:
                                            qa_text = f"{qna['prompt']}{qna['completion']}"
                                            metadata = {
                                                "group_name": selected_group["name"],
                                                "group_jid": selected_group["jid"],
                                                "prompt": qna["prompt"],
                                                "completion": qna["completion"]
                                            }
                                            batch_texts.append((qa_text, metadata))
                                        
                                        # Generate embeddings
                                        texts = [text for text, _ in batch_texts]
                                        response = openai_client.embeddings.create(
                                            model="text-embedding-ada-002",
                                            input=texts
                                        )
                                        embeddings = [v.embedding for v in response.data]
                                        
                                        # Prepare vectors for Pinecone
                                        vectors = []
                                        for j, (text, metadata) in enumerate(batch_texts):
                                            vectors.append((
                                                f"{selected_group['jid']}_{i+j}",  # unique ID
                                                embeddings[j],
                                                metadata
                                            ))
                                        
                                        # Store in Pinecone
                                        st.session_state.pinecone.upsert(vectors=vectors)
                                        total_stored += len(vectors)
                                        
                                        # Update progress
                                        progress = (i + len(batch)) / len(st.session_state.qna_pairs)
                                        st.progress(progress)
                                    
                                    st.success(f"Successfully stored {total_stored} Q&A pairs in the knowledge base!")
                                    
                                except Exception as e:
                                    st.error(f"Failed to store Q&A pairs in the knowledge base: {str(e)}")
                if st.session_state.pinecone:
                    run_rag_chatbot(st.session_state.pinecone, client)
                                
                                    
# # Footer
# st.markdown("---")
# st.markdown("*Built with ❤️ using Streamlit and OpenAI*")