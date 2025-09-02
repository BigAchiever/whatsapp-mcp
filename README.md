# WhatsApp Knowledge Miner

A Streamlit app to **extract, analyze, and generate knowledge** from WhatsApp group messages using OpenAI and Pinecone.

---

## Features

* Fetch messages from WhatsApp group chats stored in a local SQLite database.
* Analyze messages and generate structured **Q\&A pairs** using OpenAI GPT models.
* Store and update Q\&A knowledge in **Pinecone** vector database.
* Avoid duplicate entries by updating existing messages in Pinecone.
* Export generated Q\&A pairs as **JSONL**.
* Simple, interactive **Streamlit UI** with stats and message previews.

---

## Requirements

* Python 3.10+
* Streamlit
* OpenAI Python SDK (`openai`)
* Pinecone SDK (`pinecone-client`)
* SQLite3

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup

1. **Clone the repository**

```bash
git clone <repo_url>
cd whatsapp-mcp
```

2. **Create `.streamlit/secrets.toml`**

Add your API keys:

```toml
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
```

> ⚠️ Do **not** commit this file to GitHub. Add `.streamlit/secrets.toml` to your `.gitignore`.

3. **Configure Pinecone index**

The app expects a Pinecone index with the following configuration:

```python
pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
)
```

4. **Run the Streamlit app**

```bash
streamlit run streamlit-app/app.py
```

---

## Usage

1. Select a WhatsApp group from the dropdown.
2. Load messages from the SQLite DB.
3. Generate Q\&A pairs using AI.
4. Preview the top pairs and optionally store them in Pinecone.
5. Download the Q\&A pairs as JSONL for offline usage.

---

## File Structure

```
whatsapp-mcp/
├─ streamlit-app/
│  ├─ app.py                # Main Streamlit app
│  ├─ rag_chatbot.py        # RAG chatbot integration
│  └─ .streamlit/secrets.toml (API keys, excluded from GitHub)
├─ store/
│  └─ messages.db           # WhatsApp SQLite DB
├─ requirements.txt
└─ README.md
```

---

## Notes

* The app **updates messages in Pinecone** if they already exist, avoiding duplicates.
* You need your **own API keys** for OpenAI and Pinecone to use the app.
* Messages are fetched only from **group chats** (`%@g.us`).

---


If you want, I can also create a **shorter “WhatsApp-style” README** that’s super readable for non-technical users. Do you want me to do that version too?
