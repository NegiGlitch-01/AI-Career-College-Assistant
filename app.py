import streamlit as st
from google import genai
from google.genai import types
import os
import time
import random
import textwrap
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Career & College Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HTML HELPER
# =========================================================

def render_html(html):
    st.html(textwrap.dedent(html).strip())


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>

.hero {
    padding: 42px;
    border-radius: 22px;
    margin-bottom: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(80, 60, 160, 0.35),
            rgba(20, 90, 120, 0.25)
        );
    border: 1px solid rgba(255,255,255,0.10);
}

.hero-content {
    text-align: center;
}

.ai-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    background: rgba(99,102,241,0.18);
    color: #c7d2fe;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: 40px;
    font-weight: 800;
    margin-top: 18px;
}

.hero p {
    color: #a1a1aa;
    font-size: 16px;
}

.status-box {
    padding: 15px 20px;
    border-radius: 14px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.20);
    margin-bottom: 20px;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    box-shadow: 0 0 10px rgba(34,197,94,0.7);
}

.feature-card {
    padding: 22px;
    min-height: 165px;
    border-radius: 16px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-text {
    color: #a1a1aa;
    line-height: 1.6;
    font-size: 14px;
}

.welcome-box {
    padding: 22px;
    border-radius: 16px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.welcome-box p {
    color: #a1a1aa;
    line-height: 1.6;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin: 22px 0 12px 0;
}

.custom-footer {
    text-align: center;
    padding: 30px 10px;
    color: #71717a;
    font-size: 13px;
}

.stat-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

/* Professional UI enhancements */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.hero {
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 55px rgba(0,0,0,0.22);
}

.hero::before {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -70px;
    top: -100px;
    border-radius: 50%;
    background: rgba(99,102,241,0.12);
}

.ai-badge {
    border: 1px solid rgba(129,140,248,0.25);
}

.feature-card {
    transition: transform .2s ease, border-color .2s ease, background .2s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    border-color: rgba(129,140,248,0.28);
    background: rgba(255,255,255,0.055);
}

.status-box {
    box-shadow: 0 8px 28px rgba(0,0,0,0.10);
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 8px 12px;
}

[data-testid="stChatInput"] {
    border-radius: 16px;
}

.stButton > button {
    border-radius: 12px;
    min-height: 44px;
    font-weight: 650;
    transition: all .18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
}

.sidebar-card {
    padding: 14px;
    border-radius: 14px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    margin: 8px 0;
}

.source-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 650;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.20);
}

@media (max-width: 800px) {
    .hero { padding: 28px 18px; }
    .hero h1 { font-size: 30px; }
}

/* Final layout polish */
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
}

section[data-testid="stSidebar"] .stMarkdown {
    overflow-wrap: anywhere;
}

.hero {
    padding: 30px 34px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 34px;
    margin-top: 14px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 14px;
    margin-bottom: 0;
}

.status-box {
    padding: 12px 16px;
    margin-bottom: 16px;
}

.feature-card {
    padding: 16px;
    min-height: 135px;
}

.feature-icon {
    font-size: 26px;
    margin-bottom: 6px;
}

.feature-title {
    font-size: 16px;
    margin-bottom: 5px;
}

.feature-text {
    font-size: 13px;
    line-height: 1.45;
}

.section-title {
    font-size: 18px;
    margin: 16px 0 9px 0;
}

.welcome-box {
    padding: 16px;
    margin-bottom: 14px;
}

.welcome-box h3 {
    margin-top: 0;
    margin-bottom: 6px;
}

.welcome-box p {
    margin-bottom: 0;
    font-size: 13px;
}

.source-pill {
    margin-bottom: 6px;
}

.custom-footer {
    padding: 20px 10px;
}

@media (max-width: 900px) {
    .hero { padding: 24px 18px; }
    .hero h1 { font-size: 28px; }
}

</style>
""")


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "messages": [],
    "pdf_text": "",
    "pdf_chunks": [],
    "embeddings": None,
    "embedding_dimension": 0,
    "faiss_index": None,
    "pdf_name": "",
    "pdf_pages": 0,
    "document_count": 0,
    "embedding_model": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("🧠 Loading AI embedding model..."):

    embedding_model = load_embedding_model()

st.session_state.embedding_model = embedding_model


# =========================================================
# SEARCH SIMILAR CHUNKS
# =========================================================

def search_similar_chunks(question, k=3):

    faiss_index = st.session_state.faiss_index
    chunks = st.session_state.pdf_chunks
    model = st.session_state.embedding_model

    if (
        faiss_index is None
        or not chunks
        or model is None
    ):
        return []

    k = min(k, len(chunks))

    question_embedding = model.encode(
        [question],
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    distances, indices = faiss_index.search(
        question_embedding,
        k
    )

    results = []

    for index in indices[0]:

        if 0 <= index < len(chunks):

            results.append(
                chunks[index]
            )

    return results


# =========================================================
# GEMINI RESPONSE
# =========================================================

def stream_ai_answer(
    question,
    context="",
    has_document=False,
    chat_history=None
):
    """Fast Gemini response with streaming + minimal prompt/history."""

    if not GEMINI_API_KEY or client is None:
        yield "⚠️ Gemini API key is missing. Add GEMINI_API_KEY to your .env file."
        return

    # Keep the prompt small: smaller input = less work and faster response.
    recent = []
    if chat_history:
        for message in chat_history[-4:]:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            if content:
                recent.append(f"{role}: {content[:500]}")

    history_text = "\n".join(recent)
    context = (context or "")[:6000]

    if has_document and context.strip():
        prompt = f"""You are a fast, helpful AI Career & College Assistant.
Answer the user's question directly in the same language/style (English, Hindi or Hinglish).
If the question is about the uploaded document, use the document context as the primary source.
Do not invent college-specific facts. If the context does not contain the answer, say that briefly and then give general guidance if useful.
For unrelated questions, answer normally from your knowledge. Never require a PDF for general questions.
Keep answers concise unless the user asks for detail.

DOCUMENT CONTEXT:
{context}

RECENT CHAT:
{history_text}

USER QUESTION:
{question}"""
    else:
        prompt = f"""You are a fast, helpful general-purpose AI assistant.
Answer the user's question directly in the same language/style (English, Hindi or Hinglish).
You can answer questions about Python, SQL, data analytics, ML, AI, careers, resumes, interviews, college, projects, technology and general topics.
Keep answers concise unless the user asks for detail.

RECENT CHAT:
{history_text}

USER QUESTION:
{question}"""

    # Stream tokens so the user sees the answer immediately instead of waiting
    # for the entire response to finish.
    for model_name in GEMINI_MODELS:
        got_text = False
        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                ),
            )

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    got_text = True
                    yield chunk_text

            if got_text:
                return

        except Exception as e:
            error_text = str(e).lower()

            if (
                "401" in error_text
                or "403" in error_text
                or "api key" in error_text
                or "permission denied" in error_text
            ):
                yield "⚠️ Gemini API access problem. Please check your GEMINI_API_KEY."
                return

            # No sleep here. If the fast model is unavailable, immediately
            # try the fallback model so the user does not wait unnecessarily.
            if got_text:
                return

            continue

    yield "⚠️ AI service did not respond right now. Please try again." 


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:


    # =====================================================
    # SIDEBAR HEADER
    # =====================================================

    render_html("""
    <div style="
        text-align:center;
        padding:8px 0 18px 0;
    ">

        <div style="font-size:45px;">
            🤖
        </div>

        <h2 style="
            margin:5px 0;
            color:white;
        ">
            AI Assistant
        </h2>

        <p style="
            color:#9ca3af;
            font-size:13px;
        ">
            Career & College Intelligence
        </p>

    </div>
    """)


    # =====================================================
    # KNOWLEDGE BASE
    # =====================================================

    st.markdown("### 📚 Knowledge Base")


    uploaded_files = st.file_uploader(
        "Upload College PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple college PDFs."
    )


    if uploaded_files:

        st.success(
            f"✅ {len(uploaded_files)} PDF(s) selected"
        )


        for file in uploaded_files:

            st.caption(
                f"📄 {file.name} • "
                f"{file.size / 1024:.1f} KB"
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # =================================================
        # PROCESS DOCUMENTS
        # =================================================

        if st.button(
            "⚡ Process All Documents",
            use_container_width=True
        ):

            with st.spinner(
                "📚 Reading and processing documents..."
            ):

                try:

                    all_text = []

                    all_chunks = []

                    total_pages = 0


                    # -------------------------------------
                    # TEXT SPLITTER
                    # -------------------------------------

                    splitter = RecursiveCharacterTextSplitter(

                        chunk_size=800,

                        chunk_overlap=150,

                        length_function=len
                    )


                    # -------------------------------------
                    # READ ALL PDFs
                    # -------------------------------------

                    for uploaded_file in uploaded_files:


                        reader = PdfReader(
                            uploaded_file
                        )


                        total_pages += len(
                            reader.pages
                        )


                        document_text = ""


                        for page in reader.pages:

                            page_text = (
                                page.extract_text()
                                or ""
                            )

                            document_text += (
                                page_text + "\n"
                            )


                        if document_text.strip():

                            all_text.append(
                                document_text
                            )


                            chunks = (
                                splitter.split_text(
                                    document_text
                                )
                            )


                            all_chunks.extend(
                                chunks
                            )


                    # -------------------------------------
                    # CHECK CONTENT
                    # -------------------------------------

                    if not all_chunks:

                        st.error(
                            "❌ No readable text found "
                            "in the uploaded PDFs."
                        )

                    else:


                        # ---------------------------------
                        # CREATE EMBEDDINGS
                        # ---------------------------------

                        embeddings = (
                            embedding_model.encode(
                                all_chunks,
                                show_progress_bar=False
                            )
                        )


                        embedding_array = np.asarray(
                            embeddings,
                            dtype="float32"
                        )


                        # ---------------------------------
                        # CREATE FAISS
                        # ---------------------------------

                        dimension = (
                            embedding_array.shape[1]
                        )


                        faiss_index = (
                            faiss.IndexFlatL2(
                                dimension
                            )
                        )


                        faiss_index.add(
                            embedding_array
                        )


                        # ---------------------------------
                        # SAVE SESSION DATA
                        # ---------------------------------

                        st.session_state.pdf_text = (
                            "\n\n".join(all_text)
                        )

                        st.session_state.pdf_chunks = (
                            all_chunks
                        )

                        st.session_state.embeddings = (
                            embeddings
                        )

                        st.session_state.embedding_dimension = (
                            dimension
                        )

                        st.session_state.faiss_index = (
                            faiss_index
                        )

                        st.session_state.pdf_name = (
                            ", ".join(
                                file.name
                                for file in uploaded_files
                            )
                        )

                        st.session_state.pdf_pages = (
                            total_pages
                        )

                        st.session_state.document_count = (
                            len(uploaded_files)
                        )


                        # ---------------------------------
                        # SUCCESS
                        # ---------------------------------

                        st.success(
                            "✅ All documents processed!"
                        )

                        st.success(
                            f"📖 Pages: {total_pages}"
                        )

                        st.success(
                            f"📝 Chunks: {len(all_chunks)}"
                        )

                        st.success(
                            f"🧠 Embeddings: "
                            f"{len(embeddings)} × "
                            f"{dimension}"
                        )

                        st.success(
                            f"🗄️ FAISS vectors: "
                            f"{faiss_index.ntotal}"
                        )

                        time.sleep(1)

                        st.rerun()


                except Exception as e:

                    st.error(
                        f"❌ PDF processing error: {e}"
                    )


    if st.session_state.document_count:
        render_html(f"""
        <div class="sidebar-card">
            <div style="font-weight:700;">📚 Knowledge Base Loaded</div>
            <div style="color:#a1a1aa; font-size:12px; margin-top:6px;">
                {st.session_state.document_count} document(s) •
                {st.session_state.pdf_pages} page(s) •
                {len(st.session_state.pdf_chunks)} chunks
            </div>
        </div>
        """)


    # =====================================================
    # KNOWLEDGE BASE STATUS
    # =====================================================

    st.markdown("---")

    st.markdown("### 🧠 AI System")


    if st.session_state.faiss_index is not None:

        render_html("""
        <div class="status-box">

            <span class="status-dot"></span>

            AI System Online

        </div>
        """)


        st.caption("RAG Engine")

        st.progress(1.0)


        st.caption(
            f"✅ "
            f"{st.session_state.faiss_index.ntotal}"
            f" vectors indexed"
        )


        st.caption(
            f"📚 "
            f"{st.session_state.document_count}"
            f" document(s)"
        )


        st.caption(
            f"📖 "
            f"{st.session_state.pdf_pages}"
            f" page(s)"
        )


    else:

        render_html("""
        <div class="status-box">

            <span class="status-dot"></span>

            AI System Ready

        </div>
        """)


        st.caption("RAG Engine")

        st.progress(0.15)

        st.caption(
            "📄 Upload a PDF to activate RAG"
        )


    # =====================================================
    # PROJECT INFO
    # =====================================================

    st.markdown("---")

    st.markdown("### 🚀 Project")


    render_html("""
    <div class="sidebar-card">
        <div style="font-weight:700;">🤖 AI-Powered RAG Chatbot</div>
        <div style="color:#9ca3af; font-size:12px; margin-top:5px;">
            Python • Streamlit • LangChain • FAISS • NLP • Gemini
        </div>
    </div>
    """)


    # =====================================================
    # CLEAR CHAT
    # =====================================================

    st.markdown("---")


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">

    <div class="hero-content">

        <div class="ai-badge">
            ✨ AI-POWERED • RAG TECHNOLOGY
        </div>

        <h1>
            🤖 AI Career & College Assistant
        </h1>

        <p>
            Your intelligent companion for college information,
            career planning & skill development
        </p>

    </div>

</div>
""")


# =========================================================
# MAIN STATUS
# =========================================================

if st.session_state.faiss_index is not None:

    status_message = (
        "AI Assistant Online • Knowledge Base Connected"
    )

else:

    status_message = (
        "AI Assistant Ready • Ask me anything "
        "about college or career"
    )


render_html(
    f"""
    <div class="status-box">

        <span class="status-dot"></span>

        {status_message}

    </div>
    """
)


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    render_html("""
    <div class="feature-card">

        <div class="feature-icon">
            📚
        </div>

        <div class="feature-title">
            College Knowledge
        </div>

        <div class="feature-text">
            Ask about courses, exams, rules,
            eligibility, departments and
            college policies.
        </div>

    </div>
    """)


with col2:

    render_html("""
    <div class="feature-card">

        <div class="feature-icon">
            🎯
        </div>

        <div class="feature-title">
            Career Guidance
        </div>

        <div class="feature-text">
            Explore career paths, required skills,
            projects, internships and
            job preparation strategies.
        </div>

    </div>
    """)


with col3:

    render_html("""
    <div class="feature-card">

        <div class="feature-icon">
            🧠
        </div>

        <div class="feature-title">
            Smart AI Answers
        </div>

        <div class="feature-text">
            Ask questions naturally and receive
            intelligent, context-aware responses
            using Gemini.
        </div>

    </div>
    """)


# =========================================================
# PROJECT ARCHITECTURE
# =========================================================

if len(st.session_state.messages) == 0:
    render_html("""
    <div style="
        margin-top:12px;
        padding:12px 16px;
        border-radius:15px;
        background:rgba(99,102,241,0.055);
        border:1px solid rgba(129,140,248,0.13);
    ">
        <div style="font-weight:700; margin-bottom:6px;">
            ⚙️ How this assistant works
        </div>
        <div style="color:#a1a1aa; font-size:13px; line-height:1.6;">
            Your question → PDF semantic search (when available) → Gemini AI →
            streamed answer
        </div>
    </div>
    """)


# =========================================================
# WELCOME
# =========================================================

if len(st.session_state.messages) == 0:

    render_html("""
    <div class="section-title">
        💬 Start a Conversation
    </div>
    """)


    render_html("""
    <div class="welcome-box">

        <h3>
            👋 Welcome to your AI Assistant!
        </h3>

        <p>
            I can help you with college information,
            career planning, programming skills,
            projects and interview preparation.
        </p>

    </div>
    """)


# =========================================================
# QUICK QUESTIONS
# =========================================================

render_html("""
<div class="section-title">
    ⚡ Quick Questions
</div>
""")


q1, q2, q3, q4 = st.columns(4)


quick_question = None


with q1:

    if st.button(
        "📊 Data Analyst Skills",
        use_container_width=True
    ):

        quick_question = (
            "What skills are required for a Data Analyst?"
        )


with q2:

    if st.button(
        "💼 Career Roadmap",
        use_container_width=True
    ):

        quick_question = (
            "Give me a roadmap to become a Data Scientist."
        )


with q3:

    if st.button(
        "🐍 Python Projects",
        use_container_width=True
    ):

        quick_question = (
            "Suggest some good Python projects for my resume."
        )


with q4:

    if st.button(
        "🎤 Interview Prep",
        use_container_width=True
    ):

        quick_question = (
            "Give me important Data Analyst interview questions."
        )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=(
            "🤖"
            if message["role"] == "assistant"
            else "👤"
        )
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask something... e.g. What skills are required for a Data Analyst?"
)


if quick_question:

    user_input = quick_question


# =========================================================
# CHAT LOGIC
# =========================================================

if user_input:

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message(
        "user",
        avatar="👤"
    ):
        st.markdown(user_input)

    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        # -------------------------------------------------
        # SEARCH PDF ONLY IF A KNOWLEDGE BASE EXISTS
        # -------------------------------------------------

        results = []

        if st.session_state.faiss_index is not None:

            with st.spinner(
                "🔎 Checking your knowledge base..."
            ):

                results = search_similar_chunks(
                    user_input,
                    k=2
                )

        # -------------------------------------------------
        # PREPARE CONTEXT
        # -------------------------------------------------

        if results:

            context = "\n\n".join(results)

            source_message = (
                "📚 Relevant information found in "
                "your uploaded document(s)"
            )

            has_document = True

        else:

            context = ""

            source_message = (
                "🤖 General AI response"
            )

            has_document = False

        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        render_html(f'<div class="source-pill">{source_message}</div>')

        # Stream the answer so text appears as soon as Gemini starts
        # generating it. This feels much faster than waiting for the
        # complete response.
        response = st.write_stream(
            stream_ai_answer(
                question=user_input,
                context=context,
                has_document=has_document,
                chat_history=st.session_state.messages[:-1]
            )
        )

    # =====================================================
    # SAVE AI RESPONSE
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="custom-footer">

    Built by Karan Negi 👽🤷‍♂️

    <br><br>

    <strong>
        Python • Streamlit • LangChain • FAISS • NLP • Gemini AI
    </strong>

    <br><br>

    AI-Powered Career & College Assistant

</div>
""")
