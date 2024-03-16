import streamlit as st 
import Home_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler
from PyPDF2 import PdfReader
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain
import os


def get_memory(): 
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) 
    
    return memory

def get_index():
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
    ) 
    
    loader = PyPDFLoader(file_path="./ec2-gsg.pdf") 
    
    text_splitter = RecursiveCharacterTextSplitter( 
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=1000, 
        chunk_overlap=100 
    )
    
    index_creator = VectorstoreIndexCreator( 
        vectorstore_cls=FAISS, 
        embedding=embeddings, 
        text_splitter=text_splitter, 
    )
    
    index_from_loader = index_creator.from_loaders([loader]) 
    
    return index_from_loader 

def get_rag_chat_response(input_text, memory, index, streaming_callback): 
    
    model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
        model_id="anthropic.claude-v2", 
        model_kwargs=model_parameter,
        callbacks=[streaming_callback]
) 
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory)
    
    chat_response = conversation_with_retrieval({"question": input_text}) 
    
    return chat_response['answer']

st.set_page_config(page_title="Home")

if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 


for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 


input_text = st.text_input("Ask me anything if you need a support") 

if input_text:
    st_callback = StreamlitCallbackHandler(st.container())

    st.session_state.chat_history.append({"role":"user", "text":input_text}) 
    
    chat_response = get_rag_chat_response(input_text, get_memory(), get_index(), st_callback) 
    
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 