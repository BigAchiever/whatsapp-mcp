

# WhatsApp Knowledge Miner

Turn your WhatsApp **group chats** into a **searchable knowledge base**.
Messages are captured, analyzed, and converted into **Q\&A pairs** you can search using AI.

---

## Quick Start

### 1. Run the WhatsApp Bridge

This app connects to your WhatsApp account and securely stores messages in a local SQLite database.

```bash
cd whatsapp-bridge
go run main.go
```

* Scan the QR code to log in (like WhatsApp Web).
* Your messages will be saved locally in `store/messages.db`.
* **Safe:** All data stays on your machine.

---

### 2. Run the Streamlit App

This app analyzes your stored messages and lets you chat with them.

```bash
cd whatsapp-mcp
streamlit run streamlit-app/app.py
```

Open the provided URL in your browser to access the chatbot UI.

---

## ✨ Features

* 🔗 **Bridge WhatsApp → SQLite** (your messages are stored locally, no cloud).
* 🤖 **Generate Q\&A pairs** from your group chats with AI.
* 📚 **Search knowledge** using Pinecone vector database.
* 🛡️ **Avoid duplicates** (existing entries are updated, not repeated).
* 📂 **Export Q\&A pairs** as JSONL.
* 🎛️ **Simple UI** to pick groups, preview messages, and query your data.

---

## 📂 Project Structure

```
whatsapp-mcp/
├─ whatsapp-bridge/          # Go app for WhatsApp → SQLite bridge
│  └─ main.go
├─ streamlit-app/
│  ├─ app.py                 # Main Streamlit UI
│  ├─ rag_chatbot.py         # RAG chatbot logic
│  └─ .streamlit/secrets.toml (🔑 API keys, not in Git)
├─ store/
│  └─ messages.db            # Local WhatsApp messages
├─ requirements.txt
└─ README.md
```

---

## 🔑 Setup Secrets

Create `.streamlit/secrets.toml` inside `streamlit-app/`:

```toml
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
```

⚠️ Never commit this file — it should be in `.gitignore`.

---

## 🛠️ Requirements

* Python **3.10+**
* Go (for `whatsapp-bridge`)
* OpenAI API key
* Pinecone API key
* Streamlit

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Flow

1. Start **whatsapp-bridge** → authenticate WhatsApp → messages saved in `store/messages.db`.
2. Start **Streamlit app** → select group → generate & search Q\&A knowledge.
3. Export your Q\&A pairs if needed.
4. Talk with your personalized WA chatbot
