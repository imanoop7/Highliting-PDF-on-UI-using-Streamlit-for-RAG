import pandas as pd
import streamlit as st
import os
import fitz
import tempfile
from langchain.chains import RetrievalQA
import io
from streamlit_pdf_viewer import pdf_viewer
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Set page config
st.set_page_config(page_title="📚 ChatPDF")

# Main content
st.markdown(
    "<h1 style='text-align: center;'>📚 ChatPDF</h1>", unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
    
    loader = PyPDFLoader(temp_file.name)
    documents = loader.load()
    
    os.unlink(temp_file.name)  # Clean up the temporary file
    return documents


def locate_pages_containing_excerpts(document, excerpts):
    relevant_pages = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        if any(page.search_for(excerpt) for excerpt in excerpts):
            relevant_pages.append(page_num)
    return relevant_pages if relevant_pages else [0]


@st.cache_resource
def initialize_language_model():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )


@st.cache_resource
def get_embeddings():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"  # Adjust this URL if Ollama is running on a different address
    )
    return embeddings


@st.cache_resource
def setup_qa_system(documents):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        text_chunks = text_splitter.split_documents(documents)

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


def generate_highlight_annotations(document, excerpts):
    annotations = []
    for page_num, page in enumerate(document):
        for excerpt in excerpts:
            for inst in page.search_for(excerpt):
                annotations.append({
                    "page": page_num + 1,
                    "x": inst.x0,
                    "y": inst.y0,
                    "width": inst.x1 - inst.x0,
                    "height": inst.y1 - inst.y0,
                    "color": "red",
                })
    return annotations


CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user question. If you
don't know the answer, just say that you don't know, don't try to make up an
answer.

{context}

Question: {question}

Please provide your answer in the following JSON format: 
{{
    "answer": "Your detailed answer here",
    "sources": "Direct sentences or paragraphs from the context that support 
        your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
        ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
}}

The JSON must be a valid json format and can be read with json.loads() in
Python. Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file = uploaded_file.read()

    with st.spinner("Processing file..."):
        documents = extract_documents_from_file(file)
        st.session_state.doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")

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
                    parsed_result = json.loads(result['result'])

                    answer = parsed_result['answer']
                    sources = parsed_result['sources']

                    # answer = "test"
                    # sources = "Our findings indicate that the importance of science and critical thinking skills are strongly negatively associated with exposure, suggesting that occupations requiring these skills are less likely to be impacted by current LLMs. Conversely, programming and writing skills show a strong positive association with exposure, implying that occupations involving these skills are more susceptible to being influenced by LLMs."
                    sources = sources.split(". ") if pd.notna(sources) else []

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.chat_message("assistant").write(answer)

                    # Update the session state with new sources
                    st.session_state.sources = sources

                    # Set a flag to indicate chat interaction has occurred
                    st.session_state.chat_occurred = True

                except json.JSONDecodeError:
                    st.error(
                        "There was an error parsing the response. Please try again."
                    )

            # Highlight PDF excerpts
            if file and st.session_state.get("chat_occurred", False):
                doc = st.session_state.doc
                st.session_state.total_pages = len(doc)
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 0

                # Find the page numbers containing the excerpts
                pages_with_excerpts = locate_pages_containing_excerpts(doc, sources)

                if "current_page" not in st.session_state:
                    st.session_state.current_page = pages_with_excerpts[0]

                # Save the current document state in session
                st.session_state.cleaned_sources = sources
                st.session_state.pages_with_excerpts = pages_with_excerpts

                # PDF display section
                st.markdown("### PDF Preview with Highlighted Excerpts")

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
                annotations = generate_highlight_annotations(doc, st.session_state.sources)

                # Find the first page with excerpts
                if annotations:
                    first_page_with_excerpts = min(ann["page"] for ann in annotations)
                else:
                    first_page_with_excerpts = st.session_state.current_page + 1

                # Display the PDF viewer
                pdf_viewer(
                    file,
                    width=700,
                    height=800,
                    annotations=annotations,
                    pages_to_render=[first_page_with_excerpts],
                )