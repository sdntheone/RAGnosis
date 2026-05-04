# 🚀 RAGnosis – Production-Ready RAG Knowledge Assistant

RAGnosis is a high-performance Retrieval-Augmented Generation (RAG) system designed for low-latency, context-aware question answering using LLMs.  
It combines semantic search, optimized retrieval, and production-grade deployment practices to deliver fast, scalable, and reliable AI responses.

---

## 📸 Demo

![RAGnosis Chatbot Demo](assets/demo.JPG)

---

## 🔥 Key Features

- ⚡ Low-latency responses (~2–3 seconds)
- 🎯 High retrieval accuracy with optimized top-k search
- 🧠 LLM-based answer generation (OpenAI GPT)
- 🔍 Semantic search using FAISS + embeddings
- 🛠 Context pruning and pipeline optimization
- 🐳 Dockerized backend & frontend
- 🚀 CI/CD automation using GitHub Actions
- ☁️ Deployment on AWS EC2
- 📦 External vector DB mounting for scalability

---

## 🏗️ System Architecture
User Query
↓
Retriever (FAISS + Embeddings)
↓
Top-k Documents (k=2)
↓
Context Pruning & Formatting
↓
LLM (OpenAI GPT)
↓
Final Response


---

## ⚙️ Tech Stack

### Backend
- FastAPI

### Frontend
- Streamlit

### LLM & Framework
- OpenAI (gpt-4o-mini)
- LangChain

### Vector Store
- FAISS

### Embeddings
- Hugging Face (all-MiniLM-L6-v2)

### MLOps & Deployment
- Docker  
- GitHub Actions (CI/CD)  
- AWS EC2  
- MLflow (optional experiment tracking)  

---

## 📊 Performance

| Metric | Value |
|--------|------|
| Retrieval Hit Rate | 100% |
| Answer Quality | ~4.3 / 5 |
| Latency | ~2–3 seconds |

---

## ⚡ Optimization Techniques

- Reduced latency from **~50s → ~3s**
- Tuned retrieval parameter (`k=2`) for optimal context selection
- Applied **context pruning** (document truncation)
- Cached vector store and embedding model
- Eliminated blocking dependencies (MLflow at runtime)

---

## 🚀 Deployment Architecture

- Backend and frontend containerized using Docker
- CI/CD pipeline automates:
  - Build → Push → Deploy
- Deployed on AWS EC2 instance
- Containers communicate via Docker network
- Vector database mounted as external volume:
/home/ubuntu/vector_db → /app/vector_db


---

## ⚙️ Environment Variables

Create a `.env` file:

OPENAI_API_KEY=your_api_key_here


---

## 💻 Run Locally
git clone https://github.com/sdntheone/RAGnosis.git

cd RAGnosis

pip install -r requirements.txt

uvicorn main:app --reload


---

## 🐳 Run with Docker

### Build image
docker build -t ragnosis .


### Run backend
docker run -d
-p 8000:8000
--name rag_backend
--env-file .env
ragnosis


### Run frontend
docker run -d
-p 8501:8501
--name rag_frontend
-e API_URL=streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0


---

## ☁️ Run on EC2 (Production)

1. Launch EC2 instance  
2. Install Docker  
3. Pull image from Docker Hub  
4. Run containers:

docker run -d
--restart unless-stopped
--network ragnosis_network
-p 8000:8000
--name rag_backend
--env-file /home/ubuntu/.env
-v /home/ubuntu/vector_db:/app/vector_db
sdntheone/ragnosis:latest



---

## 🧠 Key Learnings

- Retrieval optimization is critical for RAG performance  
- Context pruning reduces latency and token cost  
- Production issues often stem from infrastructure (network, data, env)  
- Decoupling vector DB from container improves scalability  
- CI/CD pipelines ensure reliable deployment  

---

## 🔗 Links

- GitHub: https://github.com/sdntheone/RAGnosis