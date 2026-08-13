import streamlit as st
import requests
import os
import json

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="SPPU Syllabus QA",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #262730;
        color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #464648;
    }
    .source-box strong {
        color: #1f77b4;
    }
    .source-box em {
        color: #b0b0b0;
    }
    .extract-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #464648;
        margin: 1rem 0;
    }
    .extract-field {
        padding: 0.5rem 0;
        border-bottom: 1px solid #3a3a3d;
    }
    .extract-field:last-child {
        border-bottom: none;
    }
    .extract-label {
        color: #b0b0b0;
        font-size: 0.85rem;
    }
    .extract-value {
        color: #fafafa;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .extract-missing {
        color: #6b6b6b;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📚 SPPU Syllabus Question Answering</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload syllabus PDFs and ask questions in natural language</p>', unsafe_allow_html=True)

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload Syllabus")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload SPPU syllabus PDFs to add them to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Ingest PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    # Prepare file for upload
                    files = {
                        "file": (uploaded_file.name, uploaded_file, "application/pdf")
                    }
                    
                    # Call ingest API
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data['message']}")
                        st.info(f"Created {data['chunks']} chunks from {data['filename']}")
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Failed to connect to API: {str(e)}")
    
    st.divider()
    
    # Instructions
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Upload** a syllabus PDF using the file uploader above
    2. Click **Ingest PDF** to process it
    3. **Ask questions** in the chat interface
    4. Get **precise answers** with source citations
    """)
    
    st.divider()
    
    # Example questions
    st.header("💡 Example Questions")
    st.markdown("""
    - What are the course objectives?
    - What are the prerequisites?
    - How many credits is this course?
    - What topics are covered in Unit 1?
    - What is the exam pattern?
    """)

# Main chat interface
st.header("💬 Ask a Question")

# Main content area with tabs
tab_chat, tab_extract = st.tabs(["💬 Ask a Question", "📊 Extract Course Info"])

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and not message.get("confident", True):
                st.warning("⚠️ Low Confidence Answer")

            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}:</strong> {source['filename']}, Page {source['page_number']}<br>
                            <strong>Relevance:</strong> {(1 - source['distance']):.2%} match<br>
                            <em>{source['text_preview']}</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask about the syllabus..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"question": question}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        confident = data.get("confident", True)

                        if not confident:
                            st.warning("⚠️ Low Confidence Answer")

                        st.markdown(answer)

                        if sources:
                            with st.expander("📎 View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"""
                                    <div class="source-box">
                                        <strong>Source {i}:</strong> {source['filename']}, Page {source['page_number']}<br>
                                        <strong>Relevance:</strong> {(1 - source['distance']):.2%} match<br>
                                        <em>{source['text_preview']}</em>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "confident": confident
                        })

                    elif response.status_code == 404:
                        error_msg = "⚠️ No documents found. Please upload a PDF first."
                        st.warning(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                    else:
                        error_msg = f"❌ Error: {response.json().get('detail', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except Exception as e:
                    error_msg = f"❌ Failed to connect to API: {str(e)}\n\nMake sure the backend is running at {API_URL}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab_extract:
    st.markdown("Extract credit distribution and exam pattern as structured data for a specific course.")

    course_query = st.text_input(
        "Course name or code",
        placeholder="e.g. Natural Language Processing, or 410252"
    )

    if st.button("🔍 Extract Info", type="primary"):
        if not course_query.strip():
            st.warning("Please enter a course name or code.")
        else:
            with st.spinner("Extracting structured data..."):
                try:
                    response = requests.post(
                        f"{API_URL}/extract",
                        json={"course_query": course_query}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        data = result["data"]

                        def fmt(value, suffix=""):
                            if value is None:
                                return '<span class="extract-missing">Not found</span>'
                            return f'<span class="extract-value">{value}{suffix}</span>'

                        st.markdown(f"""
                        <div class="extract-card">
                            <h4>{data.get('course_name') or 'Unknown Course'} 
                                <span style="color:#888; font-size:0.9rem;">({data.get('course_code') or 'N/A'})</span>
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**Teaching Scheme (hrs/week)**")
                            st.markdown(f"""
                            <div class="extract-card">
                                <div class="extract-field"><span class="extract-label">Lecture</span><br>{fmt(data.get('lecture_hours_per_week'))}</div>
                                <div class="extract-field"><span class="extract-label">Practical</span><br>{fmt(data.get('practical_hours_per_week'))}</div>
                                <div class="extract-field"><span class="extract-label">Tutorial</span><br>{fmt(data.get('tutorial_hours_per_week'))}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown("**Credit Scheme**")
                            st.markdown(f"""
                            <div class="extract-card">
                                <div class="extract-field"><span class="extract-label">Theory</span><br>{fmt(data.get('theory_credits'))}</div>
                                <div class="extract-field"><span class="extract-label">Practical</span><br>{fmt(data.get('practical_credits'))}</div>
                                <div class="extract-field"><span class="extract-label">Tutorial</span><br>{fmt(data.get('tutorial_credits'))}</div>
                                <div class="extract-field"><span class="extract-label">Total</span><br>{fmt(data.get('total_credits'))}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown("**Examination Scheme (marks)**")
                            st.markdown(f"""
                            <div class="extract-card">
                                <div class="extract-field"><span class="extract-label">In-Semester (ISE)</span><br>{fmt(data.get('in_semester_exam_marks'))}</div>
                                <div class="extract-field"><span class="extract-label">End-Semester (ESE)</span><br>{fmt(data.get('end_semester_exam_marks'))}</div>
                                <div class="extract-field"><span class="extract-label">Term Work</span><br>{fmt(data.get('term_work_marks'))}</div>
                                <div class="extract-field"><span class="extract-label">Practical/Oral</span><br>{fmt(data.get('practical_oral_marks'))}</div>
                                <div class="extract-field"><span class="extract-label">Total</span><br>{fmt(data.get('total_marks'))}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        if data.get("extraction_notes"):
                            st.info(f"📝 Note: {data['extraction_notes']}")

                        with st.expander("📎 View Sources & Raw JSON"):
                            st.markdown("**Sources used:**")
                            for src in result.get("sources", []):
                                st.markdown(f"- {src['filename']}, Page {src['page_number']}")
                            st.markdown("**Raw JSON:**")
                            st.json(data)

                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Extraction failed: {error_detail}")

                except Exception as e:
                    st.error(f"❌ Failed to connect to API: {str(e)}")
# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Built with ❤️ using Streamlit, FastAPI, ChromaDB, and Ollama
</div>
""", unsafe_allow_html=True)