import streamlit as st
import json 
from langchain_groq import ChatGroq 
from langchain.schema import Document 
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_community.vectorstores import FAISS 
from sentence_transformers import SentenceTransformer 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_huggingface import HuggingFaceEmbeddings 
import os 

# Load environment variables
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

# Page config
st.set_page_config(page_title="TechX Product Chatbot", page_icon="🤖")
st.title("TechX Product Chatbot")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load the model
@st.cache_resource
def load_model():
    return ChatGroq(model="llama3-8b-8192")

# Load and process data
@st.cache_resource
def load_data():
    # Load JSON data
    with open('sample.json', 'r') as file:
        data = json.load(file)
    
    # Convert products into documents
    documents = [
        Document(
            page_content=(
                f"Model: {product['model']}\n"
                f"Brand: {product['brand']}\n"
                f"Price: ${product['price']}\n"
                f"Release Date: {product['release_date']}\n"
                f"Specifications: {product['specifications']}\n"
                f"Features: {', '.join(product['features'])}\n"
                f"Common Issues: {product['common_issues']}\n"
            ),
            metadata={"id": product["id"], "brand": product["brand"]}
        )
        for product in data["products"]
    ]
    
    # Add warranty and support documents
    warranty_doc = Document(
        page_content=(
            f"Standard Warranty: {data['warranty_info']['standard_warranty']}\n"
            f"Extended Warranty: {data['warranty_info']['extended_warranty']}\n"
            f"Coverage: {', '.join(data['warranty_info']['coverage'])}\n"
            f"Not Covered: {', '.join(data['warranty_info']['not_covered'])}\n"
        ),
        metadata={"type": "warranty"}
    )
    
    support_doc = Document(
        page_content=(
            f"Support Phone: {data['support_contacts']['phone']}\n"
            f"Support Email: {data['support_contacts']['email']}\n"
            f"Hours: {data['support_contacts']['hours']}\n"
            f"Average Response Time: {data['support_contacts']['average_response_time']}\n"
        ),
        metadata={"type": "support"}
    )
    
    documents.extend([warranty_doc, support_doc])
    return documents

# Initialize embeddings and vector store
@st.cache_resource
def initialize_retriever():
    documents = load_data()
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever()

# Load components
model = load_model()
retriever = initialize_retriever()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add this helper function to format specifications nicely
def format_product_info(doc_content):
    lines = doc_content.split('\n')
    formatted_response = ""
    
    for line in lines:
        if line.startswith('Specifications:'):
            # Convert string representation of dict to actual dict and format it
            specs_str = line.replace('Specifications:', '').strip()
            specs = eval(specs_str)  # Be careful with eval in production!
            formatted_response += "Specifications:\n"
            for key, value in specs.items():
                if isinstance(value, dict):  # Handle nested camera specs
                    formatted_response += f"• {key.replace('_', ' ').title()}:\n"
                    for cam_key, cam_val in value.items():
                        formatted_response += f"  - {cam_key}: {cam_val}\n"
                else:
                    formatted_response += f"• {key.replace('_', ' ').title()}: {value}\n"
        elif line.startswith('Features:'):
            features = line.replace('Features:', '').strip()
            formatted_response += f"\nFeatures:\n"
            for feature in features.split(','):
                formatted_response += f"• {feature.strip()}\n"
        else:
            # Handle other basic lines
            if ':' in line:
                key, value = line.split(':', 1)
                formatted_response += f"{key.strip()}: {value.strip()}\n"
    
    return formatted_response

# Modify the chat response section in your Streamlit app
if prompt := st.chat_input("Ask me about TechX products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get relevant documents
    retrieved_docs = retriever.invoke(prompt)
    
    if retrieved_docs:
        # Filter for exact model if asking about specifications
        if "specifications" in prompt.lower() and "techx pro" in prompt.lower():
            relevant_doc = next((doc for doc in retrieved_docs 
                               if "TechX Pro" in doc.page_content), None)
            
            if relevant_doc:
                formatted_response = format_product_info(relevant_doc.page_content)
                response = f"Here are the specifications for TechX Pro:\n\n{formatted_response}"
            else:
                response = "I couldn't find the specific information about TechX Pro."
        else:
            # Handle other types of queries...
            response = format_product_info(retrieved_docs[0].page_content)
    else:
        response = "I couldn't find any relevant information. Please try asking another question."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chatbot can help you with:
    - Product specifications
    - Troubleshooting
    - Warranty information
    - Support contacts
    
    Just ask your question in the chat!
    """)