import streamlit as st
from typing import List, Dict
import time

def get_query_embedding(query: str, openai_client) -> List[float]:
    """Generate embedding for the user query"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_qna(query: str, pinecone_index, openai_client, top_k: int = 5) -> List[Dict]:
    """Retrieve top-k relevant Q&A pairs from Pinecone"""
    try:
        query_vector = get_query_embedding(query, openai_client)
        if not query_vector:
            return []
            
        results = pinecone_index.query(
            vector=query_vector, 
            top_k=top_k, 
            include_metadata=True
        )
        
        qna_list = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            # Add similarity score
            metadata['similarity_score'] = match.get('score', 0)
            qna_list.append(metadata)
        return qna_list
    except Exception as e:
        st.error(f"Error retrieving knowledge: {str(e)}")
        return []

def generate_answer(query: str, retrieved_qna: List[Dict], openai_client) -> Dict[str, str]:
    """Generate a ChatGPT-style thoughtful answer using WhatsApp Q&A knowledge base."""
    
    if not retrieved_qna:
        return {
            "answer": "I couldn’t find anything in your WhatsApp knowledge base related to this question. Try rephrasing or adding more chats to the knowledge base.",
            "confidence": "low",
            "sources_used": 0
        }
    
    # Filter high-confidence matches
    high_conf_qna = [q for q in retrieved_qna if q.get('similarity_score', 0) > 0.7]
    context_qna = high_conf_qna if high_conf_qna else retrieved_qna[:3]  # fallback to top 3
    
    # Pack context more naturally
    context_text = "\n".join([
        f"- In group '{q.get('group_name','Unknown')}' on {q.get('timestamp','Unknown')}, "
        f"{q.get('sender','Unknown')} shared:\n"
        f"  Q: {q.get('prompt','').replace('Q: ','').strip()}\n"
        f"  A: {q.get('completion','').replace('A: ','').strip()}\n"
        f"  (Relevance: {q.get('similarity_score',0):.2f})"
        for q in context_qna
    ])

    prompt = f"""
You are a helpful and thoughtful assistant. 
Your job is to answer the user’s question using only the provided WhatsApp knowledge base context. 

Context from WhatsApp knowledge base:
{context_text}

User Question: {query}

Guidelines:
1. Use the context as your main evidence.
2. If multiple sources align, summarize them together.
3. If sources conflict, explain the conflict clearly.
4. If context is incomplete, say what is missing (don’t guess).
5. Be conversational, clear, and concise — like ChatGPT would.
6. Always ground answers in context (mention group/sender if helpful).
7. Keep answers concise (6-8 sentences). If more detail is helpful, add a short section with bullet points under clear headings.

Now, craft the best possible answer:
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a reasoning assistant that answers questions 
                using evidence from WhatsApp group knowledge base. 
                You synthesize, explain conflicts, and avoid hallucinations."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,  # slightly higher for natural tone
            max_tokens=600
        )
        
        answer = response.choices[0].message.content.strip()

        # Improved confidence calculation
        avg_score = sum(q.get('similarity_score', 0) for q in context_qna) / len(context_qna)
        if "not enough" in answer.lower() or "don’t know" in answer.lower() or "couldn’t find" in answer.lower():
            confidence = "low"
        elif avg_score > 0.8:
            confidence = "high"
        elif avg_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources_used": len(context_qna)
        }
        
    except Exception as e:
        return {
            "answer": f"⚠️ Error while generating answer: {str(e)}",
            "confidence": "error",
            "sources_used": 0
        }


def render_knowledge_stats():
    """Render knowledge base statistics in a compact way"""
    if 'pinecone_index' not in st.session_state:
        st.warning("🔌 **Knowledge Base:** Not connected")
        return
    
    try:
        stats = st.session_state.pinecone_index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors > 0:
            st.success(f"🧠 **Knowledge Base:** {total_vectors:,} Q&A pairs ready")
        else:
            st.warning("📭 **Knowledge Base:** Empty - Use the Extractor tab to add knowledge first")
            
    except Exception as e:
        st.error(f"⚠️ **Knowledge Base:** Connection error - {str(e)}")

def render_source_card(source: Dict, index: int) -> None:
    """Render a single source card"""
    similarity = source.get('similarity_score', 0)
    confidence_color = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🔴"
    
    with st.expander(f"{confidence_color} Source {index + 1} - {source.get('group_name', 'Unknown')} (Score: {similarity:.2f})", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📅 When:**")
            st.text(source.get('timestamp', 'Unknown time'))
            st.markdown("**👤 Who:**")
            st.text(source.get('sender', 'Unknown sender'))
        
        with col2:
            st.markdown("**💬 Group:**")
            st.text(source.get('group_name', 'Unknown group'))
            st.markdown("**🎯 Relevance:**")
            st.text(f"{similarity:.1%}")
        
        st.markdown("**❓ Original Question:**")
        st.info(source.get('prompt', '').replace('Q: ', '').strip())
        
        st.markdown("**✅ Original Answer:**")
        st.success(source.get('completion', '').replace('A: ', '').strip())

def normalize_messages(messages):
    """Convert old message format to new format"""
    normalized = []
    
    # Handle None or empty input
    if not messages:
        return normalized
    
    for message in messages:
        if "role" in message:
            # Already in new format
            normalized.append(message)
        elif "user" in message and "bot" in message:
            # Convert old format to new format
            normalized.append({
                "role": "user",
                "content": message["user"],
                "metadata": message.get("metadata", {})
            })
            normalized.append({
                "role": "assistant", 
                "content": message["bot"],
                "metadata": message.get("metadata", {})
            })
    return normalized

def render_chatbot_tab():
    """Render the enhanced ChatGPT-style chatbot interface"""

    # Custom CSS for better chat experience
    st.markdown("""
    <style>
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
    .source-counter {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    # Normalize messages
    st.session_state.messages = normalize_messages(st.session_state.messages)

    # Knowledge base status and controls
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        render_knowledge_stats()

    with col2:
        st.session_state.show_sources = st.toggle(
            "📚 Show Sources", value=st.session_state.show_sources
        )

    with col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Check if services are available
    services_available = (
        "openai_client" in st.session_state
        and "pinecone_index" in st.session_state
        and st.session_state.get("openai_client") is not None
        and st.session_state.get("pinecone_index") is not None
    )

    if not services_available:
        st.error("❌ **Required services not available.**")
        st.info("💡 Check your OpenAI and Pinecone keys in configuration.")
        return

    # Display chat messages ABOVE input
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

                metadata = message.get("metadata", {})
                confidence = metadata.get("confidence", "unknown")
                sources_used = metadata.get("sources_used", 0)

                if confidence != "error":
                    confidence_class = f"confidence-{confidence}"
                    st.markdown(
                        f"""
                        <div class="source-counter">
                            <span class="{confidence_class}">Confidence: {confidence.title()}</span>
                            &nbsp;&nbsp;📊 <strong>{sources_used}</strong> sources used
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.session_state.show_sources and metadata.get("sources"):
                    with st.expander(f"📚 View {len(metadata['sources'])} Sources", expanded=False):
                        for i, source in enumerate(metadata["sources"]):
                            render_source_card(source, i)

    # --- Input stays pinned at bottom ---
    user_input = st.chat_input("Ask me about your WhatsApp conversations...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching knowledge base..."):
                relevant_qna = retrieve_relevant_qna(
                    user_input,
                    st.session_state.pinecone_index,
                    st.session_state.openai_client,
                    top_k=7,
                )

                response_data = generate_answer(
                    user_input, relevant_qna, st.session_state.openai_client
                )

                answer = response_data["answer"]
                st.markdown(answer)

                confidence = response_data["confidence"]
                sources_used = response_data["sources_used"]

                if confidence != "error":
                    confidence_class = f"confidence-{confidence}"
                    st.markdown(
                        f"""
                        <div class="source-counter">
                            <span class="{confidence_class}">Confidence: {confidence.title()}</span>
                            &nbsp;&nbsp;📊 <strong>{sources_used}</strong> sources used
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.session_state.show_sources and relevant_qna:
                    with st.expander(f"📚 View {len(relevant_qna)} Sources", expanded=False):
                        for i, source in enumerate(relevant_qna):
                            render_source_card(source, i)

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "metadata": {
                    "confidence": confidence,
                    "sources_used": sources_used,
                    "sources": relevant_qna,
                },
            }
        )

        st.rerun()