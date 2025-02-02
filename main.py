from core import run_llm
import streamlit as st 
from streamlit_chat import message
from typing import Union, List ,Set

st.header("LangChain helper ChatBot")

if ("chat_answer_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
 ):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []

prompt  = st.text_input("prompt", placeholder="Enter you query..")

def create_source_string(source_url: Set[str]) -> str:
    if not source_url:
        return ""
    
    # Correct usage: Convert Set to list
    source_list = list(source_url)
    source_list.sort()
    
    sources_string = "sources:\n"
    for i, source in enumerate(source_list):
        sources_string += f"{i + 1}. {source}\n"
        
    return sources_string


if prompt:
    with st.spinner("Generating response"):
        generate_response = run_llm(prompt , chat_history = st.session_state["chat_history"])
        print(generate_response["result"])
        source = set([doc.metadata['source'] for doc in generate_response["source_document"]])
        fromatted_response  = (
            f"{generate_response['result']} \n\n {create_source_string(source)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(fromatted_response)
        st.session_state["chat_history"].append(("human" ,prompt))
        st.session_state["chat_history"].append(("ai" ,generate_response["result"]))
        
        
if st.session_state["user_prompt_history"]:
    for generated_response , user_query in zip(st.session_state["chat_answer_history"],st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)