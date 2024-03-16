import streamlit as st 
import Home_lib as glib 
from langchain.callbacks import StreamlitCallbackHandler
from PyPDF2 import PdfReader

st.set_page_config(page_title="Home")

if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 


for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 


uploaded_file = st.file_uploader("Upload your customer support content in PDF")
input_text = st.text_input("Ask me anything if you need a support") 

docs = []
if uploaded_file is not None and input_text:
    st_callback = StreamlitCallbackHandler(st.container())
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        docs.append(page.extract_text())

    st.session_state.chat_history.append({"role":"user", "text":input_text}) 
    
    chat_response = glib.get_rag_chat_response(input_text, docs, st_callback) 
    
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 