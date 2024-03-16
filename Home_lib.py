import os
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper

from dotenv import load_dotenv
load_dotenv()

def get_llm(streaming_callback):
        
    model_parameter = {"temperature": 0, "top_p": 0, "max_tokens_to_sample": 1000}
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
        model_id="anthropic.claude-v2", 
        model_kwargs=model_parameter,
        streaming=True,
        callbacks=[streaming_callback]
        ) 
    
    return llm
    
def get_rag_chat_response(question, context, streaming_callback): 
    llm = get_llm(streaming_callback)
    system_prompt = """System: You are a customer support assistant. Anwser in very customer obsesion style
    Only anwser related to customer support document provide in the <text> tag content provided. Say I don't know if the question does not related to provide content in <text> tag
    This is your customer support content: """ + "<text>" + str(context) + "</text>"""
    user_prompt = "User: " + question
    prompt = "System:" + system_prompt + "\n\nHuman: " + user_prompt + "\n\nAssistant:"
    return llm.invoke(prompt)