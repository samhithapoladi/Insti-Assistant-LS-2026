import os
import streamlit as st

from main_file import (
    Config,
    load_or_build_index,
    GeminiRAG,
)

st.set_page_config(page_title="IITB Insti-Assist", layout="wide")

UPLOADED_DOCS_DIR = "uploaded_docs"

CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --ib-navy: #16213E;
    --ib-navy-deep: #0D1526;
    --ib-gold: #C6A15B;
    --ib-gold-deep: #A9822E;
    --ib-bg: #090C13;
    --ib-panel: #131A2B;
    --ib-panel-raised: #171F33;
    --ib-ink: #E7EAF3;
    --ib-muted: #8D97B0;
    --ib-border: #232C42;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ib-ink);
}
.stApp {
    background: var(--ib-bg);
}
/* Hero banner */
.ib-hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: linear-gradient(135deg, var(--ib-navy) 0%, var(--ib-navy-deep) 100%);
    border: 1px solid rgba(198, 161, 91, 0.28);
    border-radius: 14px;
    padding: 1.9rem 2.2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45);
}
.ib-seal { flex-shrink: 0; }
.ib-hero-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ib-gold);
    margin-bottom: 0.35rem;
    font-weight: 600;
}
.ib-hero-title {
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 2.15rem;
    color: #FFFFFF;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}
.ib-hero-sub {
    font-size: 0.95rem;
    color: #C7CEE0;
    max-width: 660px;
    margin: 0;
    line-height: 1.55;
}
.ib-hero-sub b { color: #E9DCC2; font-weight: 600; }
/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ib-navy);
    border-right: 1px solid var(--ib-navy-deep);
}
[data-testid="stSidebar"] * { color: #E7EAF3; }
[data-testid="stSidebar"] h1 {
    font-family: 'Lora', serif;
    font-weight: 700;
    color: #FFFFFF !important;
    border-bottom: 2px solid var(--ib-gold);
    padding-bottom: 0.6rem;
    margin-bottom: 0.3rem;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.76rem !important;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ib-gold) !important;
    margin-top: 1.3rem;
    margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #A9B2C8 !important;
}
/* Card-style grouping for bordered containers in the sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(198, 161, 91, 0.35) !important;
    border-radius: 10px;
    padding: 0.3rem 0.2rem;
}
/* Sidebar form controls */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.22) !important;
}
/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: rgba(255, 255, 255, 0.06);
    color: #FFFFFF;
    border: 1px solid var(--ib-gold);
    border-radius: 7px;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--ib-gold);
    border-color: var(--ib-gold);
    color: var(--ib-navy-deep);
}
div:not([data-testid="stSidebar"]) > div > .stButton > button {
    background: var(--ib-navy);
}
/* Chat */
[data-testid="stChatMessage"] {
    background: var(--ib-panel);
    border: 1px solid var(--ib-border);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.7rem;
    color: var(--ib-ink);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(198, 161, 91, 0.07);
    border-color: rgba(198, 161, 91, 0.3);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--ib-panel-raised);
    border-left: 3px solid var(--ib-gold);
}
/* Chat input — a raised, gold-bordered search-bar feel */
[data-testid="stChatInput"] {
    background: var(--ib-panel-raised);
    border: 1.5px solid var(--ib-border);
    border-radius: 16px;
    padding: 0.15rem 0.3rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--ib-gold);
    box-shadow: 0 0 0 3px rgba(198, 161, 91, 0.18), 0 4px 18px rgba(0, 0, 0, 0.35);
}
[data-testid="stChatInput"] textarea {
    border: none !important;
    background: transparent !important;
    color: var(--ib-ink) !important;
    font-size: 0.98rem !important;
    padding: 0.65rem 0.5rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--ib-muted) !important;
}
[data-testid="stChatInput"] button {
    background: var(--ib-gold) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] button svg {
    fill: var(--ib-navy-deep) !important;
}
/* Expanders (sources) */
[data-testid="stExpander"] {
    border: 1px solid var(--ib-border);
    border-radius: 10px;
    background: var(--ib-panel);
    color: var(--ib-ink);
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--ib-gold);
}
[data-testid="stExpander"] p, [data-testid="stExpander"] span {
    color: var(--ib-ink);
}
[data-testid="stExpander"] blockquote {
    border-left: 3px solid var(--ib-gold);
    background: rgba(198, 161, 91, 0.08);
    padding: 0.65rem 1rem;
    margin: 0.5rem 0 1rem 0;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    color: var(--ib-ink);
}
[data-testid="stExpander"] blockquote p {
    margin: 0;
}
.ib-mono, [data-testid="stExpander"] code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--ib-ink) !important;
}
/* Alerts */
[data-testid="stAlert"] { border-radius: 10px; }
/* Headings in main area */
h1, h2 {
    font-family: 'Lora', serif;
    color: var(--ib-ink);
}
</style>"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


SEAL_SVG = (
    '<svg class="ib-seal" width="60" height="60" viewBox="0 0 64 64" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="32" cy="32" r="30" fill="none" stroke="#C6A15B" stroke-width="2"/>'
    '<circle cx="32" cy="32" r="24" fill="none" stroke="#C6A15B" stroke-width="1"/>'
    '<text x="32" y="39" text-anchor="middle" font-family="Lora, serif" font-size="19" '
    'font-weight="700" fill="#C6A15B">IB</text>'
    '<circle cx="32" cy="11" r="1.6" fill="#C6A15B"/>'
    '<circle cx="32" cy="53" r="1.6" fill="#C6A15B"/>'
    '<circle cx="11" cy="32" r="1.6" fill="#C6A15B"/>'
    '<circle cx="53" cy="32" r="1.6" fill="#C6A15B"/>'
    '</svg>'
)


HERO_HTML = (
    '<div class="ib-hero">'
    + SEAL_SVG +
    '<div>'
    '<div class="ib-hero-eyebrow">Institute Knowledge Assistant</div>'
    '<div class="ib-hero-title">IITB Insti-Assist</div>'
    '<p class="ib-hero-sub">Ask about hostel rules, academic policies, or council SOPs. '
    'Every answer is grounded strictly in <b>the documents you load</b>; '
    'if it isn\'t in the source material, the assistant says <b>"I don\'t know"</b> '
    'rather than guess.</p>'
    '</div>'
    '</div>'
)


with st.sidebar:
    st.title("IITB Insti-Assist")
    st.caption("A RAG-powered assistant grounded in real documents.")

    with st.container(border=True): 
        st.subheader("Gemini API Key")
        env_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API key",
            value="",
            type="password",
            placeholder="Paste your Gemini API key here or set GEMINI_API_KEY env var.",
            help="Never commit this key to a public repo.",
            label_visibility="collapsed",
        )
        api_key = api_key_input.strip() or env_key

    with st.container(border=True):
        st.subheader("Knowledge base")
        if "doc_source_mode" not in st.session_state:
            st.session_state.doc_source_mode = "folder"

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Use 'docs' folder", use_container_width=True):
                st.session_state.doc_source_mode = "folder"
        with col2:
            if st.button("📤 Upload PDFs", use_container_width=True):
                st.session_state.doc_source_mode = "upload"

        st.caption(
            f"Current source: **{'docs folder' if st.session_state.doc_source_mode == 'folder' else 'uploaded PDFs'}**"
        )

        if st.session_state.doc_source_mode == "folder":
            doc_dir = st.text_input("Documents folder", value="docs")
        else:
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                help="Uploaded files are saved locally to the "
                     f"'{UPLOADED_DOCS_DIR}/' folder and indexed from there.",
            )
            doc_dir = UPLOADED_DOCS_DIR

            if uploaded_files:
                os.makedirs(doc_dir, exist_ok=True)
                saved_names = []
                for uf in uploaded_files:
                    save_path = os.path.join(doc_dir, uf.name)
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_names.append(uf.name)
                st.success(f"Saved {len(saved_names)} file(s) to '{doc_dir}/'.")

            if os.path.isdir(doc_dir):
                existing = sorted(
                    f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf")
                )
                if existing:
                    with st.expander(f"📂 {len(existing)} PDF(s) currently in '{doc_dir}/'"):
                        for name in existing:
                            st.text(name)

        top_k = Config.TOP_K
        min_sim = Config.MIN_SIMILARITY

        rebuild = st.checkbox("Force rebuild index", value=False)
        build_clicked = st.button("🔧 Build / Load Index", use_container_width=True)

    st.divider()
    st.caption("""
    **How to use:**

    1. Pick a docs folder or upload PDFs above, then click **Build / Load Index**.

    2. Ask a question in the chat box.

    3. The assistant answers only from what you loaded. It says "I don't know" if the answer isn't in there.
    """)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, content, sources}


if build_clicked:
    if not os.path.isdir(doc_dir):
        st.sidebar.error(f"Folder '{doc_dir}' not found.")
    else:
        with st.spinner("Processing documents (chunking + embedding)..."):
            try:
                st.session_state.vector_store = load_or_build_index(
                    doc_dir=doc_dir, force_rebuild=rebuild
                )
                st.sidebar.success(
                    f"Index ready — {len(st.session_state.vector_store.chunks)} chunks indexed."
                )
            except Exception as e:
                st.sidebar.error(f"Failed to build index: {e}")


def render_sources(sources):
    with st.expander("📄 Exact source text used"):
        for s in sources:
            page_str = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**{s['source']}{page_str}**")
            # Markdown blockquote — styled via CSS as a highlighted citation block
            quoted = "\n".join(f"> {line}" for line in s["text"].splitlines() or [""])
            st.markdown(quoted)
            st.markdown("---")


st.markdown(HERO_HTML, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

query = st.chat_input("Enter question.")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            answer_text = (
                "The knowledge base hasn't been built yet. Please add documents "
                "(folder or upload) and click **Build / Load Index** in the sidebar first."
            )
            st.warning(answer_text)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text, "sources": []}
            )
        elif not api_key:
            answer_text = (
                "No Gemini API key found. Paste one in the sidebar, or set the "
                "`GEMINI_API_KEY` environment variable, then try again."
            )
            st.warning(answer_text)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text, "sources": []}
            )
        else:
            with st.spinner("Retrieving relevant documents and generating an answer..."):
                try:
                    rag = GeminiRAG(
                        vector_store=st.session_state.vector_store,
                        api_key=api_key,
                        top_k=top_k,
                        min_similarity=min_sim,
                    )
                    result = rag.answer(query)
                except Exception as e:
                    result = {
                        "answer": f"Error while generating an answer: {e}",
                        "sources": [],
                        "grounded": False,
                    }

            st.markdown(result["answer"])
            if result["sources"]:
                render_sources(result["sources"])
            elif not result["grounded"]:
                st.info("No result.")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )