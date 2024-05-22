# ChatBot that uses RAG to answer question about documents.
Within this repository we implement a simple chatbot that can answer question about the documentation of the Italian Camera dei Deputati.

### Technologies used
The chatbot is build using LangChain, in particular we uses:
- docx2txt, to open and read the documents
- Faiss, to store the documents' embeddings
- HuggingFace, for both embeddings and llm
- other LangChain functionalities (text splitters, chains, etc.)

## What package do you need to run the chatbot?
- ipykernel (to run the notebook)
- langchain
- faiss-cpu
- sentence-transformers

## How to run the chatbot?
Clone the repository, insert you HugginFace token in "rag_llm.ipynb" and run it.

## Future
We are going to documentthe code more accuratly.
We are realizing a dashboard (using streamlit) to interact with the chatbot easly.
