import streamlit as st
from st_social_media_links import SocialMediaIcons
import time
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableParallel
from langchain_core.output_parsers.string import StrOutputParser

st.set_page_config(page_title="Lucio chatbot", page_icon=":fox_face:") #, layout="wide"


#-------------------------------------------------------------------------------
# VAR DI AMBIENTE E VAR GLOBALI

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "Your_Gemini_Key"

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your_HuggingFace_Key"

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "Your_OpenAI_Key"
    
api_keys = {
    "Mistral" : os.getenv('HUGGINGFACEHUB_API_TOKEN'),
    "Gemini" : os.getenv('GOOGLE_API_KEY'),
    "GPT (coming soon...)" : os.getenv('OPENAI_API_KEY')
}

chat_hist = [] # rende la char history disponibile alla chain dell'LLM


#-------------------------------------------------------------------------------
# FUNZIONI

def reset_chat_hist():
    chat_hist = []
    st.session_state.messages = []

def extract_text(x):
    documents = ""
    sources = []
    for doc in x['context']:
        documents = documents + "<chunk>" + doc.page_content + "</chunk>\n"
        sources.append(doc.metadata["source"])

    sources = list(set(sources))
    all_source = ", ".join(sources)

    if show_sources == "Visualizza i nomi documenti":
        with st.expander("Documenti utilizzati"):
          st.markdown("*I documenti utilizzati per rispondere sono: "+all_source+"*")
          st.write("\n")
    elif show_sources == "Visualizza il contenuto dei documenti":
        with st.expander("Documenti utilizzati"):
          st.markdown("*I chunks utilizzati per rispondere sono: "+documents+"*")
          st.write("\n")

    return {"context": documents, "question": x['question']}

def extract_history(x):
    history = ""
    if len(chat_hist)==0:
      return history
    for mess in chat_hist:
        if mess["role"]=="Utente":
            history = history + "Utente: " + mess["content"] + "\n"
        elif mess["role"]=="Lucio":
            history = history + "Chatbot: " + mess["content"] + "\n"
    return history


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.1)

def create_llms(llm_name, temp_llm):
    if llm_name == "Gemini":
      llm_summarize = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, top_p=0.5)
      llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temp_llm, top_p=0.5)
    elif llm_name == "Mistral":
      llm_summarize = HuggingFaceEndpoint(repo_id ='mistralai/Mixtral-8x7B-Instruct-v0.1', temperature=0.3, token=api_keys["Mistral"])
      llm = HuggingFaceEndpoint(repo_id ='mistralai/Mixtral-8x7B-Instruct-v0.1', temperature=temp_llm, token=api_keys["Mistral"])
    elif llm_name == "GPT (coming soon...)":
      llm_summarize = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_keys["GPT (coming soon...)"])
      llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temp_llm, openai_api_key=api_keys["GPT (coming soon...)"])

    return llm_summarize, llm


def create_prompts():
    prompt_resumer_template='''\
    Riscrivi la domanda dell'utente riportata tra <q> e </q> \
    utilizzando lo storico della conversazione tra utente e chatbot \
    riportata tra <conv> e </conv>. Contestualizza la domanda dell'utente \
    usando lo storico, non devi riassumere la conversazione e non devi \
    menzionare il chabot.
    <q>{question}</q>
    <conv>{history}</conv>
    '''
    prompt_template = '''\
    Sei un chatbot che deve rispondere alle domande degli utenti riguardo \
    persone, aziende ed associazioni riportate nella documentazione.
    La domanda dell'utente è riportata tra <q> e </q>, mentre una lista di
    estratti della documentazione sono riportati tra <ctx> e </ctx>, ogni \
    estratto è separato da <chunk> e </chunk>. Se la domanda dell'utente \
    non riguarda un estratto della documetazione dici che non puoi rispondere.
    <q>{question}</q>
    <ctx>{context}</ctx>
    '''
    prompt_resumer = PromptTemplate.from_template(prompt_resumer_template)
    prompt_llm = PromptTemplate.from_template(prompt_template)

    return prompt_resumer, prompt_llm



#-------------------------------------------------------------------------------
# SIDEBAR

with st.sidebar:
    st.header("Chi è Lucio?")


    description = '''Lucio è un chatbot italiano in grado di rispondere alle domande \
        degli utenti riguardante la documentazione della Camera dei Deputati italiana.\
            \nA volte è un po' timido, quindi siate gentili con lui!\n
    '''
    st.markdown(description)
    st.divider()

    st.header("Settings")
    number_of_doc = st.number_input("Numero di documenti (RAG)", min_value=1, max_value=30, value=5)
    show_sources = st.selectbox("Impostazioni di visualizzazione RAG", options=["Solo chat","Visualizza i nomi documenti","Visualizza il contenuto dei documenti"], index=0)
    st.subheader("Impostazioni LLM")
    llm_name = st.radio("Scegli il tuo LLM", ["Gemini", "Mistral", "GPT (coming soon...)"], on_change=reset_chat_hist)
    temp_llm = st.slider("Temperatura", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
    c1,c2 = st.columns(2)
    with c1:
      if st.button("Reset pagina"):
          st.rerun()
          reset_chat_hist()
    with c2:
      if st.button("Reset Lucio"):
          reset_chat_hist()

    social_media_links = [
      "https://github.com/zurbojack/rag-chatbot-I3N",
      "https://www.linkedin.com/company/langchain/",
      "https://www.youtube.com/watch?v=G1IbRujko-A",
    ]

    social_media_icons = SocialMediaIcons(social_media_links)
    social_media_icons.render()


#-------------------------------------------------------------------------------
# CARICAMENTO DATABASE E CREAZIONE LLM E RAG CHAIN

with st.spinner("Svegliando Lucio..."):

    db_faiss = FAISS.load_local("./faiss_db/", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    retriever = db_faiss.as_retriever(search_kwargs={"k": number_of_doc})

    llm_summarize, llm = create_llms(llm_name, temp_llm)
    prompt_resumer, prompt_llm = create_prompts()

    def pritn(x):
      st.write(x)
      return x

    q_summarize_chain =  RunnableBranch(
        (lambda x: len(chat_hist)==1, lambda x: x),
        lambda x: {"history": extract_history, "question": RunnablePassthrough()} | prompt_resumer | llm_summarize | StrOutputParser(),
    )
    retr_chain = RunnableParallel({"context": retriever, "question" : RunnablePassthrough()})
    rag_chain = q_summarize_chain | retr_chain | extract_text | prompt_llm | llm | StrOutputParser()



#-------------------------------------------------------------------------------
# SEZIONE PRINCIPALE

st.title("Chatta con Lucio :speech_balloon:")

# prendo l'input dell'utente
prompt = st.chat_input("Cosa ti interessa sapere?")
with st.container(height=650, border=False):

    # Inizializzo chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Mostro la chat history passata
    for message in st.session_state.messages:
        if message["role"] == "Utente":
            with st.chat_message(message["role"], avatar="🗣️"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar="🦊"):
                st.markdown(message["content"])

    if prompt:
        # Aggiungo il nuovo messaggio dell'utente alla history
        st.session_state.messages.append({"role": "Utente", "content": prompt})
        # Mostro i messaggi dell'utente nel container
        with st.chat_message("Utente", avatar="🗣️"):
            st.markdown(prompt)

        chat_hist = st.session_state.messages # serve momentaneamete per far vedere la history a extract_history nella chain
        # Mostro i messaggi del chatbot nel container
        with st.chat_message("Lucio", avatar="🦊"):
            resp = rag_chain.invoke(prompt)
            response = st.write_stream(response_generator(resp))
        # Aggiungo il nuovo messaggio del chatbot alla history
        st.session_state.messages.append({"role": "Lucio", "content": response})