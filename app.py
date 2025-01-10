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

# Load the model
model = ChatGroq(model="llama3-8b-8192")

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

# Add to document list
documents.extend([warranty_doc, support_doc])


# Load a pre-trained embedding model
model_name = "all-MiniLM-L6-v2"  # Compact and efficient model
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create vector store and retriever
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

# Command-Line Interface
def main():
    print("Welcome to the Product Query CLI!")
    print("Type your query or type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        print("\nSearching for relevant documents...\n")
        retrieved_docs = retriever.invoke(query)
        
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, start=1):
                print(f"--- Retrieved Document {i} ---\n")
                print(doc.page_content)
                print("\n")
        else:
            print("No relevant documents found. Please try another query.")

if __name__ == "__main__":
    main()