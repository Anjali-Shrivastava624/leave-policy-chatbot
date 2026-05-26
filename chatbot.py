import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import chromadb
import re
import os

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Leave Policy Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Leave Policy AI Assistant")
st.caption("Powered by RAG + LangGraph + ChromaDB + Groq (Llama 3)")

# ── Config ─────────────────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.08
PDF_PATH = r"D:\Aarav Projects\Chatbot\Leave Policy.pdf"

# ── Groq Client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Clean PDF text ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\n\s{0,3}\n', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'● ', '\n• ', text)
    return text.strip()

# ── Extract keywords from query ────────────────────────────────────────────────
def extract_query_keywords(question: str) -> List[str]:
    stopwords = {
        "how", "many", "much", "what", "is", "are", "the", "a", "an",
        "do", "does", "i", "get", "have", "can", "tell", "me", "about",
        "for", "in", "on", "of", "to", "and", "or", "my", "your", "their",
        "there", "leave", "policy", "days", "number", "total", "give"
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    keywords = [w for w in words if w not in stopwords]
    return keywords if keywords else words

# ── Sentence-level extraction ──────────────────────────────────────────────────
def extract_relevant_sentences(text: str, keywords: List[str]) -> str:
    """
    How it works:
    - Split the complete chunk into sentences
    - Check each sentence for query keywords
    - Return only matching sentences
    - If nothing matches, return the full chunk (fallback)
    """
    # Split on sentence boundaries: period, exclamation, or question mark
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    matched = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # If any keyword exists in the sentence, keep it
        if any(kw in sentence_lower for kw in keywords):
            matched.append(sentence.strip())

    return ' '.join(matched) if matched else text

# ── Re-rank chunks by keyword presence ────────────────────────────────────────
def keyword_rerank(docs: List[str], scores: List[float], keywords: List[str]) -> List[tuple]:
    ranked = []
    for doc, score in zip(docs, scores):
        doc_lower = doc.lower()
        keyword_bonus = 0.0
        for kw in keywords:
            # Count how many times the keyword appears — add bonus score per match
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', doc_lower))
            keyword_bonus += count * 0.15
        # Apply penalty if the primary keyword is completely missing from the chunk
        primary_keyword = keywords[0] if keywords else ""
        if primary_keyword and primary_keyword not in doc_lower:
            keyword_bonus -= 0.3
        ranked.append((doc, score + keyword_bonus))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# ── Groq: Generate answer from context ────────────────────────────────────────
def generate_answer_with_groq(question: str, context: str) -> str:
    """
    When relevant content is found in the PDF,
    use Groq to generate a clean and conversational
    answer based on the provided context.
    """
    system_prompt = """You are a helpful HR assistant for Aarav Solutions.
You answer employee questions strictly based on the Leave Policy document provided.

Rules:
- Answer ONLY what is asked — do not include unrelated leave types or policies
- Be concise and clear
- If the context does not contain the answer, say so honestly
- Do not make up information
- Do not introduce yourself with any name
- If asked your name, say you are the Leave Policy Assistant
- Format numbers and lists clearly"""

    user_prompt = f"""Leave Policy context:
{context}

Employee question: {question}

Answer based only on the above context:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ── Groq: Answer out-of-scope questions ───────────────────────────────────────
def answer_general_with_groq(question: str) -> str:
    """
    When the question is not found in the PDF,
    Groq answers using its general HR knowledge.
    """
    system_prompt = """You are a helpful HR assistant for Aarav Solutions.
The employee has asked a question not found in the Leave Policy document.
Answer helpfully using general HR knowledge. If unsure, say so politely.
Do not introduce yourself with any name.
If asked your name, say you are the Leave Policy Assistant."""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ── Load PDF → ChromaDB ────────────────────────────────────────────────────────
@st.cache_resource
def init_system(pdf_path: str):
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found: {pdf_path}")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(pages)
    texts = [clean_text(c.page_content) for c in chunks]

    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(texts).toarray().tolist()

    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("leave_policy")
    except Exception:
        pass

    collection = client.create_collection(
        name="leave_policy",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=texts,
        embeddings=matrix,
        ids=[f"chunk_{i}" for i in range(len(texts))],
        metadatas=[{"page": chunks[i].metadata.get("page", 0)} for i in range(len(chunks))]
    )
    return collection, vectorizer, len(texts)

# ── LangGraph State ────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    question: str
    retrieved_docs: List[str]
    scores: List[float]
    answer: str
    source: str   # "pdf_groq" | "general_groq" | "error"

# ── Retrieve Node ──────────────────────────────────────────────────────────────
def retrieve_node(state: ChatState) -> ChatState:
    query_vec = st.session_state.vectorizer.transform(
        [state["question"]]
    ).toarray().tolist()

    results = st.session_state.collection.query(
        query_embeddings=query_vec,
        n_results=5,
        include=["documents", "distances"]
    )

    docs      = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    scores    = [round(1 - d, 4) for d in distances]

    return {**state, "retrieved_docs": docs, "scores": scores}

# ── Answer Node ────────────────────────────────────────────────────────────────
def answer_node(state: ChatState) -> ChatState:
    docs     = state["retrieved_docs"]
    scores   = state["scores"]
    question = state["question"]

    try:
        # ── Case 1: Relevant content found in PDF ─────────────────────────────
        if docs and scores and scores[0] >= RELEVANCE_THRESHOLD:
            keywords = extract_query_keywords(question)

            # Chunk-level re-ranking based on keyword presence
            ranked    = keyword_rerank(docs, scores, keywords)
            best_docs = [doc for doc, sc in ranked[:2] if sc > 0] or [docs[0]]

            # Sentence-level filtering — keep only relevant sentences
            filtered = []
            seen     = set()
            for doc in best_docs:
                relevant = extract_relevant_sentences(doc, keywords)
                if relevant.strip() and relevant not in seen:
                    seen.add(relevant)
                    filtered.append(relevant.strip())

            context = '\n\n'.join(filtered)

            # Generate a clean answer using Groq
            answer = generate_answer_with_groq(question, context)

            return {**state, "answer": answer, "source": "pdf_groq"}

        # ── Case 2: Not found in PDF → use Groq general knowledge ─────────────
        else:
            answer = answer_general_with_groq(question)
            return {**state, "answer": answer, "source": "general_groq"}

    except Exception as e:
        # API error handling
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            msg = " Invalid Groq API key. Please check your GROQ_API_KEY."
        elif "rate_limit" in error_msg.lower():
            msg = " Groq rate limit reached. Please wait and try again."
        else:
            msg = f" Error from Groq: {error_msg}"
        return {**state, "answer": msg, "source": "error"}

# ── Build LangGraph ────────────────────────────────────────────────────────────
@st.cache_resource
def build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer",   answer_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer",   END)
    return workflow.compile()

# ── Initialize once per session ───────────────────────────────────────────────
if "initialized" not in st.session_state:
    with st.spinner("Loading PDF into ChromaDB... Please wait."):
        col, vec, total = init_system(PDF_PATH)
        st.session_state.collection  = col
        st.session_state.vectorizer  = vec
        st.session_state.graph       = build_graph()
        st.session_state.messages    = []
        st.session_state.initialized = True
    st.success(f"System ready! {total} chunks indexed.")

# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle user input ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about leave policy..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke({
                "question":       prompt,
                "retrieved_docs": [],
                "scores":         [],
                "answer":         "",
                "source":         ""
            })

        answer = result["answer"]
        source = result["source"]

        st.markdown(answer)

        # Source caption
        if source == "pdf_groq":
            st.caption("📄 Source: Leave_Policy.pdf  |  Answer by Groq (Llama 3)")
        elif source == "general_groq":
            st.caption("🤖 Not in PDF — Answered by Groq (Llama 3) general knowledge")
        else:
            st.caption(" Error occurred")

    st.session_state.messages.append({"role": "assistant", "content": answer})