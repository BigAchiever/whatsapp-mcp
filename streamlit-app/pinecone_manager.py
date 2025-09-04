from datetime import datetime
import time
import streamlit as st

def store_in_knowledge_base(selected_group):
    """Store Q&A pairs in Pinecone knowledge base"""
    if not hasattr(st.session_state, 'extractor_qna_pairs') or not st.session_state.extractor_qna_pairs:
        st.error("❌ No Q&A pairs available to store. Generate Q&A pairs first.")
        return
    
    with st.spinner("Storing Q&A pairs in knowledge base..."):
        try:
            # Generate embeddings and store in batches
            batch_size = 100  # more optimal
            total_stored = 0
            progress_bar = st.progress(0)
            
            for i in range(0, len(st.session_state.extractor_qna_pairs), batch_size):
                batch = st.session_state.extractor_qna_pairs[i:i + batch_size]
                batch_texts = []
                
                # Prepare texts for embedding
                for qna in batch:
                    qa_text = f"{qna['prompt']}{qna['completion']}"
                    metadata = {
                        "group_name": selected_group["name"],
                        "group_jid": selected_group["jid"],
                        "prompt": qna["prompt"],
                        "completion": qna["completion"],
                        "timestamp": datetime.now().isoformat()
                    }
                    batch_texts.append((qa_text, metadata))
                
                # Generate embeddings
                texts = [text for text, _ in batch_texts]
                response = st.session_state.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                embeddings = [v.embedding for v in response.data]
                
                # Prepare vectors for Pinecone (no timestamp in ID)
                vectors = []
                for j, (text, metadata) in enumerate(batch_texts):
                    vectors.append((
                        f"{selected_group['jid']}_{i+j}",  # unique ID without hashing/time
                        embeddings[j],
                        metadata
                    ))
                
                # Store in Pinecone
                st.session_state.pinecone_index.upsert(vectors=vectors)
                total_stored += len(vectors)
                
                # Update progress
                progress = (i + len(batch)) / len(st.session_state.extractor_qna_pairs)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            st.success(f"✅ Successfully stored {total_stored} Q&A pairs in the knowledge base!")
            
        except Exception as e:
            st.error(f"❌ Failed to store Q&A pairs: {str(e)}")
