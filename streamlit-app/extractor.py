import streamlit as st
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import time
from sklearn.metrics.pairwise import cosine_similarity
from pinecone_manager import store_in_knowledge_base
import numpy as np
import re

DB_PATH = "../whatsapp-bridge/store/messages.db"

def normalize_text(s: str) -> str:
    return " ".join(s.split()).strip().lower()

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities like dates, times, URLs, emails, phone numbers, etc."""
    entities = {
        'dates': re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}|\d{1,2}(?:st|nd|rd|th)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december))\b', text.lower()),
        'times': re.findall(r'\b(?:\d{1,2}:\d{2}(?:\s*(?:am|pm|AM|PM))?|\d{1,2}\s*(?:am|pm|AM|PM))\b', text),
        'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'phones': re.findall(r'(?:\+91|91)?[6-9]\d{9}', text),
        'numbers': re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text),
        'currencies': re.findall(r'(?:₹|Rs\.?|INR|USD|\$)\s*\d+(?:,\d{3})*(?:\.\d+)?[KMB]?', text, re.IGNORECASE),
        'percentages': re.findall(r'\d+(?:\.\d+)?%', text)
    }
    return {k: v for k, v in entities.items() if v}

def is_meaningful_message(message: str, entities: Dict[str, List[str]]) -> bool:
    """Check if message contains meaningful content worth extracting"""
    if not message or len(message.strip()) < 10:
        return False
    
    # Skip common casual phrases
    casual_phrases = [
        'good morning', 'good afternoon', 'good evening', 'good night',
        'thanks', 'thank you', 'welcome', 'ok', 'okay', 'yes', 'no',
        'sure', 'fine', 'great', 'awesome', 'cool', 'nice', 'congrats',
        'happy birthday', 'all the best', 'take care', 'bye', 'see you'
    ]
    
    lower_msg = message.lower()
    if any(phrase in lower_msg and len(lower_msg) < 30 for phrase in casual_phrases):
        return False
    
    # Consider meaningful if has entities or substantial content
    has_entities = any(entities.values())
    has_substantial_content = len(message.split()) > 8
    
    # Check for meaningful keywords
    meaningful_keywords = [
        'meeting', 'event', 'workshop', 'session', 'conference', 'webinar',
        'project', 'deadline', 'requirement', 'solution', 'proposal',
        'investment', 'funding', 'startup', 'business', 'company',
        'team', 'hire', 'job', 'opportunity', 'collaboration',
        'product', 'service', 'feature', 'launch', 'release',
        'announcement', 'update', 'news', 'information'
    ]
    
    has_meaningful_keywords = any(keyword in lower_msg for keyword in meaningful_keywords)
    
    return has_entities or has_substantial_content or has_meaningful_keywords

def is_trivial_qna(qna: Dict[str, Any]) -> bool:
    """Enhanced heuristics to drop trivial Q&As"""
    question = qna.get("prompt", "").replace("Q:", "").strip()
    answer = qna.get("completion", "").replace("A:", "").strip()
    
    if not question or not answer:
        return True
    
    # Check for very short answers without meaningful content
    if len(answer) < 15:
        return True
    
    # Check for generic/vague answers
    generic_patterns = [
        r'^(yes|no|maybe|sure|okay|ok|fine|great|good)\.?$',
        r'^(i don\'t know|not sure|will check|let me check)\.?$',
        r'^(thanks|thank you|welcome)\.?$'
    ]
    
    if any(re.match(pattern, answer.lower()) for pattern in generic_patterns):
        return True
    
    # Keep if has specific entities or detailed information
    entities = extract_entities(answer)
    has_specific_info = any(entities.values()) or len(answer.split()) > 6
    
    return not has_specific_info

def deduplicate_qna_pairs_global(qna_pairs: List[Dict[str, Any]],
                                 openai_client,
                                 threshold: float = 0.88,
                                 batch_size: int = 200) -> List[Dict[str, Any]]:
    """
    Deduplicate a list of Q&A pairs:
      - removes exact duplicate text
      - filters trivial Q&As
      - gets batched embeddings, then greedily keeps only Q&As that are < threshold similar
    Returns the filtered list.
    """
    if not qna_pairs:
        return []

    # 1) exact-string de-dupe and normalize
    seen = set()
    uniq = []
    for q in qna_pairs:
        text = normalize_text(q.get("prompt","") + " " + q.get("completion",""))
        if text in seen:
            continue
        seen.add(text)
        if is_trivial_qna(q):
            continue
        uniq.append(q)

    if not uniq:
        return []

    # 2) get texts for embeddings
    texts = [normalize_text(q.get("prompt","") + " " + q.get("completion","")) for q in uniq]

    # 3) batch embeddings
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([r.embedding for r in resp.data])
        except Exception as e:
            st.warning(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            # Skip this batch if embeddings fail
            continue

    if len(embeddings) != len(uniq):
        # If some embeddings failed, only keep the ones we have
        uniq = uniq[:len(embeddings)]

    # 4) greedy semantic dedupe using normalized vectors
    embs = np.array(embeddings, dtype=np.float32)
    # normalize each vector to unit length
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit_embs = embs / norms

    kept = []
    kept_units = []

    for i, unit in enumerate(unit_embs):
        if not kept_units:
            kept.append(uniq[i])
            kept_units.append(unit)
            continue
        # compute dot product with kept units -> cosine similarity
        sims = np.dot(np.vstack(kept_units), unit)  # shape (len(kept),)
        if sims.max() < threshold:
            kept.append(uniq[i])
            kept_units.append(unit)
        # else skip as duplicate / near-duplicate

    return kept

def get_api_key(key_name: str) -> str:
    """Get API key from secrets or environment variables"""
    try:
        return st.secrets[key_name]
    except:
        return os.getenv(key_name)

def get_sender_context(messages: List[Dict], current_index: int, window: int = 3) -> str:
    """Get context from surrounding messages to better understand the conversation"""
    start = max(0, current_index - window)
    end = min(len(messages), current_index + window + 1)
    
    context_messages = []
    for i in range(start, end):
        if i == current_index:
            continue
        msg = messages[i]
        sender = "You" if msg['is_from_me'] else (msg['sender'] or "Unknown")
        if msg['message']:
            context_messages.append(f"{sender}: {msg['message'][:100]}")
    
    return "\n".join(context_messages[-3:])  # Last 3 messages for context

def generate_qna_pairs(messages: List[Dict[str, Any]], message_index: int, openai_client) -> List[Dict[str, str]]:
    """Generate Q&A pairs from a message using GPT-4 with enhanced context"""
    message = messages[message_index]
    
    try:
        # Skip if message is not meaningful
        entities = extract_entities(message['message'])
        if not is_meaningful_message(message['message'], entities):
            return []
        
        # Get conversation context
        context = get_sender_context(messages, message_index, window=3)
        
        # Format sender information
        sender_name = "You" if message['is_from_me'] else (message['sender'] or "Unknown User")
        
        message_context = f"""
        CURRENT MESSAGE:
        From: {sender_name}
        Time: {message['timestamp']}
        Content: {message['message']}
        Media: {"Yes - " + message['media_type'] if message['media_type'] else "No"}
        
        CONVERSATION CONTEXT (recent messages):
        {context if context else "No recent context available"}
        """
        
        prompt = f"""You are a knowledge extraction expert specializing in WhatsApp group conversations. Your task is to create high-quality Q&A pairs that preserve important context and details.

ANALYZE THIS MESSAGE:
{message_context}

EXTRACTION RULES:
1. CONTEXT PRESERVATION: Always include WHO said/did something when relevant
2. ENTITY EXTRACTION: Capture ALL specific details (dates, times, URLs, names, numbers, locations, etc.)
3. CONVERSATIONAL INTELLIGENCE: Use conversation context to provide complete answers
4. QUALITY OVER QUANTITY: Only extract meaningful, knowledge-worthy content
5. USER PERSPECTIVE: Frame questions as a general user would ask a chatbot

SKIP IF MESSAGE CONTAINS ONLY:
- Simple greetings, acknowledgments, or casual responses
- Pure emotional reactions (just emojis, "lol", "thanks", etc.)
- Incomplete information without context

INCLUDE WHEN MESSAGE HAS:
- Events, meetings, or announcements with details
- Business opportunities, requirements, or proposals
- Technical information or explanations
- Contact information or links
- Deadlines, schedules, or important dates
- Company, product, or service information

FORMAT GUIDELINES:
- Questions should be natural and conversational
- Answers should be complete with all relevant context
- Include sender information when it adds value
- Preserve specific details (URLs, dates, contacts)
- Make answers self-contained (don't assume prior knowledge)

OUTPUT FORMAT:
Return ONLY valid JSON objects, one per line (JSONL format):
{{"prompt": "Q: [natural question]\\n\\n", "completion": "A: [complete contextual answer including who, what, when, where relevant]\\n"}}

EXAMPLES:
Instead of: "What is the date?" → "What date was mentioned?"
Better: "Q: When is the ABAIA mixer session scheduled?" → "A: John mentioned the ABAIA mixer session is scheduled for September 3, 2025, at 8 PM on MS Teams."

If no meaningful knowledge can be extracted, return empty."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a knowledge extraction expert for WhatsApp conversations. 
                Focus on creating high-quality, contextual Q&A pairs that preserve important details and sender context.
                Only extract meaningful information that would be valuable in a knowledge base."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent extraction
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()
        if not content:
            return []
        
        qna_pairs = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                qna = json.loads(line)
                if "prompt" in qna and "completion" in qna:
                    # Add metadata for better tracking
                    qna["metadata"] = {
                        "sender": sender_name,
                        "timestamp": message['timestamp'],
                        "message_index": message_index
                    }
                    qna_pairs.append(qna)
            except json.JSONDecodeError:
                continue

        return qna_pairs

    except Exception as e:
        st.error(f"Error generating Q&A pairs: {str(e)}")
        return []

def qna_pairs_to_jsonl(qna_pairs: List[Dict]) -> str:
    """Convert list of Q&A dicts to JSONL string, excluding metadata"""
    lines = []
    for qna in qna_pairs:
        # Create clean version without metadata for export
        clean_qna = {
            "prompt": qna.get("prompt", ""),
            "completion": qna.get("completion", "")
        }
        lines.append(json.dumps(clean_qna, ensure_ascii=False))
    return "\n".join(lines)

def get_groups():
    """Fetch distinct group JIDs with names from DB"""
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT m.chat_jid, c.name, MAX(m.timestamp) as last_active
            FROM messages m
            LEFT JOIN chats c ON m.chat_jid = c.jid
            WHERE m.chat_jid LIKE '%@g.us'
            GROUP BY m.chat_jid
            ORDER BY last_active DESC
        """)
        
        groups = []
        for row in cursor.fetchall():
            chat_jid = row[0]
            group_name = row[1] if row[1] else f"Group {chat_jid[-8:]}"
            
            groups.append({
                "jid": chat_jid,
                "name": group_name,
                "last_active": row[2]
            })
        
        # groups.sort(key=lambda x: x["name"])
        
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
        WHERE chat_jid = ? AND content IS NOT NULL AND content != ''
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (group_jid, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "timestamp": r[2], "is_from_me": r[3], "media_type": r[4], "filename": r[5]} for r in rows]
def render_extractor_tab():
    """Render the message extraction tab"""
    
    # Initialize session state for extractor with unique keys
    if 'extractor_messages' not in st.session_state:
        st.session_state.extractor_messages = None
    if 'extractor_current_group' not in st.session_state:
        st.session_state.extractor_current_group = None
    if 'extractor_qna_pairs' not in st.session_state:
        st.session_state.extractor_qna_pairs = []
    if 'extractor_show_count' not in st.session_state:
        st.session_state.extractor_show_count = 10
    if 'extractor_show_meaningful_only' not in st.session_state:
        st.session_state.extractor_show_meaningful_only = True
    if 'extractor_similarity_threshold' not in st.session_state:
        st.session_state.extractor_similarity_threshold = 0.88
    if 'extractor_max_messages_to_process' not in st.session_state:
        st.session_state.extractor_max_messages_to_process = 500
    
    # Check database connection
    if not os.path.exists(DB_PATH):
        st.error(f"⚠️ Database not found at: {DB_PATH}")
        st.info("Please ensure your WhatsApp database is properly connected.")
        return
    
    groups = get_groups()
    
    if not groups:
        st.warning("⚠️ No groups found in database.")
        st.info("Make sure your WhatsApp database contains group messages.")
        return
    
    # Group selection section
    st.subheader("📱 Select WhatsApp Group")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_group = st.selectbox(
            "Choose a group to analyze:",
            options=groups,
            format_func=lambda g: f"{g['name']}", 
            index=0,
            help="Select the WhatsApp group you want to extract messages from",
            key="extractor_group_selectbox"
        )
    
    with col2:
        message_limit = st.number_input(
            "Message limit", 
            min_value=50, 
            max_value=1000, 
            value=200, 
            step=50,
            key="extractor_message_limit"
        )
        
        if st.button("🔄 Load Messages", type="primary", use_container_width=True, key="extractor_load_messages_btn"):
            if selected_group:
                with st.spinner("Fetching messages..."):
                    st.session_state.extractor_messages = get_group_messages(selected_group["jid"], limit=message_limit)
                    st.session_state.extractor_current_group = selected_group["jid"]
                    st.session_state.extractor_selected_group_info = selected_group
                    st.session_state.extractor_qna_pairs = []  # Reset Q&A pairs when loading new messages
    
    # Group info display
    if selected_group:
        with st.container():
            st.info(f"**Selected Group:** {selected_group['name']}")
    
    # Display messages and controls if loaded
    if st.session_state.extractor_messages and st.session_state.extractor_current_group == selected_group["jid"]:
        messages = st.session_state.extractor_messages
        
        # Statistics section
        st.divider()
        st.subheader("📊 Statistics")
        
        # Calculate enhanced stats
        total_messages = len(messages)
        media_messages = len([m for m in messages if m.get('media_type')])
        unique_senders = len(set([m['sender'] for m in messages if not m['is_from_me']]))
        meaningful_messages = len([m for m in messages if is_meaningful_message(m.get('message', ''), extract_entities(m.get('message', '')))])
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", f"{total_messages:,}")
        with col2:
            st.metric("Meaningful", f"{meaningful_messages:,}")
        with col3:
            st.metric("Media Messages", f"{media_messages:,}")

        
        # Messages preview section
        st.divider()
        st.subheader("💬 Message Preview")
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.extractor_show_count = st.slider(
                    "Number of messages to preview", 
                    min_value=5, 
                    max_value=50, 
                    value=st.session_state.extractor_show_count,
                    key="extractor_show_count_slider"
                )
            with col2:
                st.session_state.extractor_show_meaningful_only = st.checkbox(
                    "Show meaningful only", 
                    value=st.session_state.extractor_show_meaningful_only,
                    key="extractor_show_meaningful_cb"
                )
            
            display_messages = (
                [m for m in messages if is_meaningful_message(m.get('message', ''), extract_entities(m.get('message', '')))]
                if st.session_state.extractor_show_meaningful_only else messages
            )[:st.session_state.extractor_show_count]
            
            for i, msg in enumerate(display_messages):
                sender_display = "You" if msg['is_from_me'] else (msg['sender'] or "Unknown")
                meaningful = is_meaningful_message(msg.get('message', ''), extract_entities(msg.get('message', '')))
                
                status_icon = "🔍" if meaningful else "💬"
                
                with st.expander(f"{status_icon} Message {i+1} - {sender_display}", expanded=False):
                    if msg['message']:
                        st.markdown(f"**Message:** {msg['message']}")
                    else:
                        st.markdown("*No text content*")
                    
                    if msg.get('media_type'):
                        st.info(
                            f"📎 **Media:** {msg['media_type']}" + 
                            (f" | **File:** {msg['filename']}" if msg.get('filename') else "")
                        )
        
        # AI Knowledge Generation section
        st.divider()
        st.subheader("🤖 AI Knowledge Generation")
        
        col1, col2 = st.columns([2, 1])
        
     
        st.markdown("Transform your messages into structured knowledge using enhanced AI analysis with context preservation.")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            st.session_state.extractor_similarity_threshold = st.slider(
                "Similarity threshold for deduplication", 
                0.70, 0.95, 
                st.session_state.extractor_similarity_threshold, 
                0.01,
                key="extractor_similarity_slider"
            )
            include_context = st.checkbox(
                "Include conversation context", 
                value=True,
                key="extractor_include_context_cb"
            )
            st.session_state.extractor_max_messages_to_process = st.number_input(
                "Max messages to process", 
                min_value=1, 
                max_value=len(messages), 
                value=min(st.session_state.extractor_max_messages_to_process, len(messages)),
                key="extractor_max_messages_input"
            )

    
        if st.button("🔍 Generate Q&A", type="primary", use_container_width=True, key="extractor_generate_qna_btn"):
            if 'openai_client' not in st.session_state:
                st.error("OpenAI client not initialized. Please check your API key.")
                return
            
            with st.spinner("Analyzing messages with AI..."):
                all_qna_pairs = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process only meaningful messages
                messages_to_process = messages[:st.session_state.extractor_max_messages_to_process]
                processed = 0
                
                for i, message in enumerate(messages_to_process):
                    if message.get('message'):
                        entities = extract_entities(message['message'])
                        if is_meaningful_message(message['message'], entities):
                            status_text.text(f"Processing meaningful message {processed+1}...")
                            qna_pairs = generate_qna_pairs(messages_to_process, i, st.session_state.openai_client)
                            if qna_pairs:
                                all_qna_pairs.extend(qna_pairs)
                            processed += 1
                    progress_bar.progress((i + 1) / len(messages_to_process))
                
                # Deduplicate Q&A pairs
                before = len(all_qna_pairs)
                if all_qna_pairs:
                    filtered = deduplicate_qna_pairs_global(all_qna_pairs, st.session_state.openai_client, 
                                                            threshold=st.session_state.extractor_similarity_threshold, batch_size=200)
                    after = len(filtered)
                    st.session_state.extractor_qna_pairs = filtered
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Generated {before:,} Q&As → deduplicated to {after:,} unique Q&As from {processed} meaningful messages")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.info("ℹ️ No knowledge-worthy content found in the analyzed messages")
    
        # Enhanced Q&A Pairs section
        if st.session_state.extractor_qna_pairs:
            st.divider()
            st.subheader("📝 Generated Q&A Pairs")
            
            # Quality metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_q_length = np.mean([len(qna['prompt']) for qna in st.session_state.extractor_qna_pairs])
                st.metric("Avg Q Length", f"{avg_q_length:.1f}")
            with col2:
                avg_a_length = np.mean([len(qna['completion']) for qna in st.session_state.extractor_qna_pairs])
                st.metric("Avg A Length", f"{avg_a_length:.1f}")
            with col3:
                unique_senders_in_qa = len(set([qna.get('metadata', {}).get('sender', 'Unknown') for qna in st.session_state.extractor_qna_pairs]))
                st.metric("Contributors", f"{unique_senders_in_qa}")
            with col4:
                st.metric("Total Q&As", f"{len(st.session_state.extractor_qna_pairs)}")
            
            # Preview section with enhanced display
            with st.expander(f"👁️ Preview Q&A Pairs ({len(st.session_state.extractor_qna_pairs)} total)", expanded=True):
                preview_count = min(5, len(st.session_state.extractor_qna_pairs))
                
                for i in range(preview_count):
                    qna = st.session_state.extractor_qna_pairs[i]
                    metadata = qna.get('metadata', {})
                    
                    st.markdown(f"**Q&A Pair {i+1}**")
                    st.code(f"Question: {qna['prompt'].replace('Q: ', '').strip()}", language="text")
                    st.code(f"Answer: {qna['completion'].replace('A: ', '').strip()}", language="text")
                    
                    if i < preview_count - 1:
                        st.divider()
                
                if len(st.session_state.extractor_qna_pairs) > preview_count:
                    st.info(f"Showing first {preview_count} pairs. Download to see all {len(st.session_state.extractor_qna_pairs)} pairs.")
            
            # Actions section
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download JSONL button
                jsonl_content = qna_pairs_to_jsonl(st.session_state.extractor_qna_pairs)
                st.download_button(
                    label="💾 Download Q&A JSONL",
                    data=jsonl_content,
                    file_name=f"whatsapp_qna_{selected_group['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
                    mime="application/json",
                    use_container_width=True,
                    key="extractor_download_jsonl_btn"
                )
            
            with col2:
                # Download with metadata
                full_content = json.dumps(st.session_state.extractor_qna_pairs, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📊 Download with Metadata",
                    data=full_content,
                    file_name=f"whatsapp_qna_full_{selected_group['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="extractor_download_full_btn"
                )
            
            with col3:
                # Store in knowledge base
                if st.session_state.get('pinecone_index'):
                    if st.button("🗄️ Store in Knowledge Base", type="secondary", use_container_width=True, key="extractor_store_kb_btn"):
                        store_in_knowledge_base(selected_group)
                else:
                    st.button("🗄️ Knowledge Base Unavailable", disabled=True, use_container_width=True, key="extractor_kb_unavailable_btn")
                    st.caption("Pinecone not configured")