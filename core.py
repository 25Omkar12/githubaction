from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from typing import Any, Dict, List
from langchain.chains.history_aware_retriever import create_history_aware_retriever
load_dotenv()

INDEX_NAME = "chatbot2"

def run_llm(query:str, chat_history: List[Dict[str, Any]]=[]):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME , embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key="AIzaSyABRx-AaiGx5YY4voEaJrQi8yFbHRw3pKQ",
        temperature=0,
        top_p=0.95,
        top_k=40,
        max_tokens=2048,
        verbose=True
    )
    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain  = create_stuff_documents_chain(llm , retrival_qa_chat_prompt)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriver = create_history_aware_retriever(llm , retriever=docsearch.as_retriever() , prompt= rephrase_prompt)
    qa = create_retrieval_chain(
        retriever=history_aware_retriver , combine_docs_chain= stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result":result["answer"],
        "source_document": result["context"],
    }
    return new_result
if __name__ == "__main__":
    res = run_llm(query = "Give the important dates in the contract provided")
    print(res["result"])