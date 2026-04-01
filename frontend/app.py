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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source['filename']} (Chunk {source['chunk_index']})<br>
                        <strong>Relevance:</strong> {(1 - source['distance']):.2%} match<br>
                        <em>{source['text_preview']}</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Ask about the syllabus..."):
    import json
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get answer from API with streaming
    with st.chat_message("assistant"):
        # Use a list to store sources (mutable, so it can be modified in generator)
        sources_container = []
        
        def response_generator():
            """Generator function for streaming response."""
            try:
                response = requests.post(
                    f"{API_URL}/chat/stream",
                    json={"question": question},
                    stream=True
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                
                                if data['type'] == 'sources':
                                    # Store sources in the container
                                    sources_container.clear()
                                    sources_container.extend(data['sources'])
                                elif data['type'] == 'token':
                                    yield data['content']
                                elif data['type'] == 'done':
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    yield f"Error: {response.status_code}"
            
            except Exception as e:
                yield f"Failed to connect to API: {str(e)}"
        
        # Stream the response
        full_response = st.write_stream(response_generator())
        
        # Display sources (they're now in sources_container)
        if sources_container:
            with st.expander("📎 View Sources"):
                for i, source in enumerate(sources_container, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source['filename']}, Page {source['page_number']}<br>
                        <strong>Relevance:</strong> {(1 - source['distance']):.2%} match<br>
                        <em>{source['text_preview']}</em>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources_container.copy()  # Copy the list
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Built with ❤️ using Streamlit, FastAPI, ChromaDB, and Ollama
</div>
""", unsafe_allow_html=True)