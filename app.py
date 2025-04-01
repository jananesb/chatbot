import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to check if image contains "Check Your Knowledge"
def is_unwanted_image(image_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img_path = tmp_img.name
        img = Image.open(tmp_img_path)
        text = pytesseract.image_to_string(img)
        return "Check Your Knowledge" in text
    except:
        return False

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        images = page.get_images(full=True)
        images_per_page[page_num] = []

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]

            # Apply filtering conditions to remove unwanted images
            if (
                width < 100  # Ignore small images (headers/logos)
                or height < 50
                or is_unwanted_image(image_bytes)  # OCR-based filtering
            ):
                continue  # Skip unwanted images

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                images_per_page[page_num].append(tmp_img.name)
    
    doc.close()
    return text_per_page, images_per_page

# Function to index PDF text with page metadata
def index_pdf_text(text_per_page):
    documents = []
    for page_num, text in text_per_page:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num})
            documents.append(doc)
    vector_store = FAISS.from_documents(documents, embedding_function)
    return vector_store

# Function to query Gemini API with concise prompt
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Valid model as of March 2025
        response = model.generate_content(
            f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise answer suitable for exam preparation."
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Function to search PDF and answer with images
def search_pdf_and_answer(query, vector_store, images_per_page):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)
    
    # Identify pages containing relevant text
    page_nums = set(doc.metadata['page'] for doc in docs)
    
    # Filter images based on relevance
    relevant_images = []
    for page_num in page_nums:
        if page_num in images_per_page:
            relevant_images.extend(images_per_page[page_num])
    
    return answer, relevant_images

# Streamlit UI
st.title("📄 PDF Chatbot with Gemini API 🤖")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    with st.spinner("Processing PDF... Please wait..."):
        text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
        vector_store = index_pdf_text(text_per_page)
    st.success("PDF successfully indexed! ✅")
    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("Generating response..."):
            answer, relevant_images = search_pdf_and_answer(query, vector_store, images_per_page)
        st.write("### 🤖 Answer:")
        st.write(answer)
        if relevant_images:
            st.write("#### Relevant Images from PDF:")
            for img_path in relevant_images:
                st.image(img_path, use_column_width=True)
