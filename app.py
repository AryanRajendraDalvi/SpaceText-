import streamlit as st
import time
import os

# Set up the page layout
st.set_page_config(page_title="SpaceText", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")

# ── BACKEND INTEGRATION ──────────────────────────────────────────────────────
# We use @st.cache_resource so the 3GB model is only loaded into RAM once, 
# not every time you click a button on the webpage!
@st.cache_resource(show_spinner="Booting up Param-1 (2.9B) Engine...")
def load_agent():
    # Attempt to import the agent. If it fails, return None so UI can show an error.
    try:
        from spacetext.rag.agent import SpaceTextAgent
        return SpaceTextAgent(
            model_path="./models/param-1-2.9b-instruct-q8_0.gguf", 
            db_path="./aerospace_taxonomy.adk"
        )
    except Exception as e:
        st.error(f"Failed to load backend: {e}")
        return None

agent = load_agent()

# Custom CSS to closely match the mockup styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95%; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .primary-header { color: #0f172a; font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .sub-header { color: #64748b; font-size: 1.1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── LEFT SIDEBAR (Navigation) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 SpaceText")
    st.caption("Aerospace Document Summarizer")
    st.markdown("---")
    
    st.radio("Menu", ["🏠 Summarize", "💬 Ask Question", "📚 Browse by Source", "📄 Document Library", "⚙️ Model & Settings"], label_visibility="collapsed")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**Sources**")
    st.markdown("🛰️ **ISRO**")
    st.markdown("🌎 **NASA**")
    st.markdown("🌌 **ESA**")
    st.markdown("🛡️ **DRDO**")
    st.markdown("📄 **ArXiv**")

# ── MAIN WORKSPACE ──────────────────────────────────────────────────────────
main_col, right_col = st.columns([2.8, 1.2], gap="large")

with main_col:
    st.markdown('<div class="primary-header">Summarize Aerospace Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a research document or search GarudDB to get a structured summary.</div>', unsafe_allow_html=True)
    
    # Input modes: File Upload OR Database Search
    search_query = st.text_input("🔍 Search GarudDB directly (RAG Pipeline):", placeholder="e.g., inertial navigation systems")
    uploaded_file = st.file_uploader("📄 Or drag and drop a raw file here:", type=["txt"], help="Currently supporting TXT. PDF parser coming soon.")
    
    # Action Button
    if st.button("Summarize", type="primary"):
        if agent is None:
            st.error("Backend agent is not loaded.")
        elif uploaded_file is not None:
            with st.spinner("Analyzing document with Param-1 (2.9B)..."):
                content = uploaded_file.getvalue().decode("utf-8")
                # Call our summarize_document method!
                st.session_state['last_result'] = agent.summarize_document(content, filename=uploaded_file.name)
        elif search_query:
            with st.spinner("Searching GarudDB and generating synthesis..."):
                # Call the RAG pipeline! The Router Agent will handle this automatically.
                st.session_state['last_result'] = agent.summarize(topic=search_query)
        else:
            st.warning("Please enter a search query or upload a file.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Structured Summary", "📖 Narrative Synthesis", "</> Raw Output"])
    
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        
        with tab1:
            if "structured_summary" in res and res["structured_summary"]:
                st.markdown(res["structured_summary"])
            else:
                st.info("Structured summary not available.")
                
        with tab2:
            if "narrative_summary" in res and res["narrative_summary"]:
                st.write(res["narrative_summary"])
            else:
                st.info("Narrative summary not available.")
                
        with tab3:
            st.json(res)
    else:
        with tab1:
            st.info("Results will appear here after summarization.")

# ── RIGHT SIDEBAR (Settings & Samples) ──────────────────────────────────────
with right_col:
    st.markdown("### ⚙️ Model & Settings")
    st.selectbox("Model", ["Param-1 (2.9B Instruct)"])
    st.selectbox("Retrieval Source", ["GarudDB (All Sources)"])
    st.slider("Max Output Length", 100, 2048, 1024, 128)
    
    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    
    # These buttons will instantly populate the search and run it!
    if st.button("Summarize ISRO satellite missions", use_container_width=True):
        with st.spinner("Running sample query..."):
            st.session_state['last_result'] = agent.summarize(topic="ISRO satellite missions")
            st.rerun()
            
    if st.button("What are DRDO's navigation systems?", use_container_width=True):
        with st.spinner("Running sample query..."):
            st.session_state['last_result'] = agent.summarize(topic="DRDO inertial navigation systems")
            st.rerun()
    
    st.markdown("---")
    st.info("**ℹ️ About SpaceText**\n\nSpaceText is a domain-specific small language model designed to understand, categorize and summarize aerospace research documents. It uses Param-1 (2.9B) with GarudDB for accurate, source-aware summaries.")
