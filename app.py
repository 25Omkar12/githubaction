from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings



def upload_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

def chunking(pages):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Max characters per chunk
        chunk_overlap=100  # Overlap between chunks
    )
    chunks = text_splitter.split_documents(pages)
    
    return chunks

if __name__ == "__main__":
    print(" Chat bot")
    document = upload_document("/Users/omkaranilmestry/Desktop/omkar/chatbot/SERVICE AGREEMENT.pdf")
    chunks = chunking(document)
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    PineconeVectorStore.from_documents(
        chunks , embeddings , index_name = "chatbot2"
    )
    
    