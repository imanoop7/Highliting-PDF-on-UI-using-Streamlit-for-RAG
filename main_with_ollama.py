import pandas as pd
import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
import io
from streamlit_pdf_viewer import pdf_viewer
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
import pymupdf4llm

# Set page config
st.set_page_config(page_title="📚 ChatPDF")

# Main content
st.markdown(
    "<h1 style='text-align: center;'>📚 ChatPDF</h1>", unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as temp_file:
        # Write the contents of the uploaded file to the temporary file
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        # Now that the file is closed, we can safely process it
        md_text = pymupdf4llm.to_markdown(temp_file_path)
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        return documents, md_text
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def locate_pages_containing_excerpts(md_text, excerpts):
    lines = md_text.split('\n')
    relevant_pages = []
    current_page = 1
    for i, line in enumerate(lines):
        if line.startswith('# Page'):
            current_page = int(line.split()[2])
        if any(excerpt in line for excerpt in excerpts):
            relevant_pages.append(current_page - 1)  # Convert to 0-based index
    return relevant_pages if relevant_pages else [0]


@st.cache_resource
def initialize_language_model():
    return ChatOllama(
        model="tinyllama",
        temperature=0,
        base_url="http://localhost:11434"
    )


@st.cache_resource
def get_embeddings():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"  # Adjust this URL if Ollama is running on a different address
    )
    return embeddings


@st.cache_resource
def setup_qa_system(_documents):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        text_chunks = text_splitter.split_documents(_documents)  # Note the underscore

        vector_store = FAISS.from_documents(text_chunks, get_embeddings())
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        )

        return RetrievalQA.from_chain_type(
            initialize_language_model(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT},
        )
    except Exception as e:
        st.error(f"Error setting up QA system: {str(e)}")
        return None


def generate_highlight_annotations(md_text, excerpts):
    annotations = []
    lines = md_text.split('\n')
    current_page = 1
    y_position = 0
    for line in lines:
        if line.startswith('# Page'):
            current_page = int(line.split()[2])
            y_position = 0
        else:
            for excerpt in excerpts:
                if excerpt in line:
                    annotations.append({
                        "page": current_page,
                        "x": 50,  # Arbitrary x position
                        "y": y_position,
                        "width": 500,  # Arbitrary width
                        "height": 20,  # Arbitrary height
                        "color": "red",
                    })
            y_position += 20  # Increment y position for each line

    return annotations


CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user question. If you
don't know the answer, just say that you don't know, don't try to make up an
answer.

{context}

Question: {question}

IMPORTANT: You must provide your answer in the following JSON format: 
{{
    "answer": "Your detailed answer here",
    "sources": "Direct sentences or paragraphs from the context that support 
        your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
        ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
}}

The JSON must be a valid json format and can be read with json.loads() in
Python. Do not include any text before or after the JSON object. Only return the JSON object. Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file = uploaded_file.read()

    with st.spinner("Processing file..."):
        documents, md_text = extract_documents_from_file(uploaded_file)
        st.session_state.md_text = md_text

    if documents:
        with st.spinner("Setting up QA system..."):
            qa_system = setup_qa_system(documents)
            if qa_system is None:
                st.error("Failed to set up QA system. Please check if Ollama is running and try again.")
            else:
                st.success("QA system ready!")
                # Continue with the rest of your chat logic here

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! How can I assist you today? "}
            ]

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input("Your message"):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            st.chat_message("user").write(user_input)

            with st.spinner("Generating response..."):
                try:
                    result = qa_system.invoke({"query": user_input})
                    
                    # Print raw result for debugging
                    st.write("Raw result:", result)
                    
                    try:
                        parsed_result = json.loads(result['result'])
                        answer = parsed_result['answer']
                        sources = parsed_result['sources']
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat the entire result as the answer
                        answer = result['result']
                        sources = ""  # No sources available in this case
                    
                    sources = sources.split(". ") if isinstance(sources, str) else []

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.chat_message("assistant").write(answer)

                    # Update the session state with new sources
                    st.session_state.sources = sources

                    # Set a flag to indicate chat interaction has occurred
                    st.session_state.chat_occurred = True

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again or check the model's configuration.")

            # Highlight PDF excerpts
            if uploaded_file and st.session_state.get("chat_occurred", False):
                st.session_state.total_pages = md_text.count('# Page')
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 0

                # Find the page numbers containing the excerpts
                pages_with_excerpts = locate_pages_containing_excerpts(md_text, sources)

                if "current_page" not in st.session_state:
                    st.session_state.current_page = pages_with_excerpts[0]

                # Save the current document state in session
                st.session_state.cleaned_sources = sources
                st.session_state.pages_with_excerpts = pages_with_excerpts

                # Just before the PDF display section, add this debug statement
                st.write("Debug: Reached PDF display section")

                # PDF display section
                st.markdown("### PDF Preview with Highlighted Excerpts")

                # Add this debug statement
                st.write(f"Debug: Total pages: {st.session_state.total_pages}")

                # Navigation
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("Previous Page") and st.session_state.current_page > 0:
                        st.session_state.current_page -= 1
                with col2:
                    st.write(
                        f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}"
                    )
                with col3:
                    if (
                        st.button("Next Page")
                        and st.session_state.current_page
                        < st.session_state.total_pages - 1
                    ):
                        st.session_state.current_page += 1

                # Get annotations with correct coordinates
                annotations = generate_highlight_annotations(md_text, st.session_state.sources)

                # Add this debug statement
                st.write(f"Debug: Number of annotations: {len(annotations)}")

                # Find the first page with excerpts
                if annotations:
                    first_page_with_excerpts = min(ann["page"] for ann in annotations)
                else:
                    first_page_with_excerpts = st.session_state.current_page + 1

                # Add this debug statement
                st.write(f"Debug: First page with excerpts: {first_page_with_excerpts}")

                try:
                    # Add this debug statement
                    st.write(f"Debug: Attempting to display PDF")
                    
                    # Create a temporary file to save the uploaded PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    # Display the PDF viewer
                    pdf_viewer(
                        temp_file_path,
                        width=700,
                        height=800,
                        annotations=annotations,
                        pages_to_render=[first_page_with_excerpts],
                    )
                    
                    # Add this debug statement
                    st.write("Debug: PDF viewer called successfully")
                except Exception as e:
                    st.error(f"Error displaying PDF: {str(e)}")
                finally:
                    # Clean up the temporary file
                    if 'temp_file_path' in locals():
                        os.unlink(temp_file_path)

                # Add this final debug statement
                st.write("Debug: End of PDF display section")
