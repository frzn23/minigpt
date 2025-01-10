--- Important ---
Add Your Groq API Key in .env file.
GROQ_API_KEY="_here_"

--------- Sample.json ----------

Detailed product information for three smartphones:

Technical specifications
Features
Common issues and their solutions
Pricing and release dates


Warranty information
Support contact details

This dataset would allow your chatbot to answer questions like:

What are the specifications of the TechX Pro?
How can I fix battery drain on my TechX Pro?
What's covered under warranty?
What's the difference between TechX Pro and TechX Lite?
How do I contact support?

Source : Generated using ClaudeAI


--------- Langchain & Groq----------
pip install langchain
pip install -U langchain-community
pip install groq
pip install -qU langchain-groq
pip install faiss-cpu langchain[docarray] # For document retrieval

We will be using Groq AI LLM pairing with Langchain

---------- ENV VARIABLES-------

pip install python-dotenv

--------- For text embeddings -----
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install sentence-transformers
