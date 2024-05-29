import os
import re
from docx import Document as word_doc
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "Your_Gemini_Key"

# ------------------------------------------
# FUNZIONI

def get_gext(document):
    full_text = []
    for para in document.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def split_into_paragraphs(text):
    # Usa espressioni regolari per suddividere il testo in paragrafi
    paragraphs = re.split(r'\n\s*\n', text)
    # Rimuovi eventuali paragrafi vuoti
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    return paragraphs


# ------------------------------------------
# SEZIONE PRINCIPALE

documents = []

for doc_name in os.listdir("./docx/"):
    paragraphs_as_doc = []
    if doc_name != "pdfs":
        document = word_doc("./docx/"+doc_name)
        full_text = get_gext(document)
        paragraphs = split_into_paragraphs(full_text)
        for para in paragraphs:
            paragraphs_as_doc.append(Document(page_content=para, metadata={"source": doc_name}))

    documents = documents + paragraphs_as_doc


gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db_faiss = None
for i in range(0, len(documents), 50):
  if db_faiss is None:
    db_faiss = FAISS.from_documents(documents[i:i+50], gemini_embeddings)
  else:
    db_faiss.merge_from( FAISS.from_documents(documents[i:i+50], gemini_embeddings) )

db_faiss.save_local('./faiss_db')

# ------------------------------------------
# TEST

retriever = db_faiss.as_retriever(search_kwargs={"k": 20})
retrieved = retriever.invoke("chi è antonucci marco?")