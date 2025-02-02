import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document
from app import upload_document, chunking
from core import run_llm
import os 


@patch("app.PyPDFLoader")  # Mock PyPDFLoader
def test_upload_document(mock_loader):
    """Test that upload_document loads and returns document pages."""
    # Mock PyPDFLoader instance
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load_and_split.return_value = [
        Document(page_content="First page text"),
        Document(page_content="Second page text"),
    ]

    file_path = "dummy.pdf"  # No actual file needed
    pages = upload_document(file_path)

    assert isinstance(pages, list)
    assert len(pages) == 2
    assert pages[0].page_content == "First page text"
    assert pages[1].page_content == "Second page text"


def test_chunking():
    """Test that chunking correctly splits documents into smaller chunks."""
    mock_documents = [
        Document(page_content="This is a long text that needs to be split. " * 20),  # Long enough to be chunked
        Document(page_content="Another long document that needs chunking. " * 20),
    ]

    chunks = chunking(mock_documents)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert len(chunks) > 2  # Should create multiple chunks if splitting works


import pytest
from unittest.mock import Mock, patch
from core import run_llm

# Mock response data
MOCK_EMBEDDING_VECTOR = [0.1] * 1024
MOCK_DOCS = [
    "Document 1 content",
    "Document 2 content"
]
MOCK_LLM_RESPONSE = {
    "input": {"input": "test query", "chat_history": []},
    "answer": "This is a test answer",
    "context": MOCK_DOCS
}

@pytest.fixture
def mock_dependencies():
    """Fixture to set up all necessary mocks"""
    with patch('core.CohereEmbeddings') as mock_embeddings, \
         patch('core.PineconeVectorStore') as mock_pinecone, \
         patch('core.ChatGoogleGenerativeAI') as mock_llm, \
         patch('core.hub') as mock_hub, \
         patch('core.create_stuff_documents_chain') as mock_stuff_chain, \
         patch('core.create_history_aware_retriever') as mock_history_retriever, \
         patch('core.create_retrieval_chain') as mock_retrieval_chain:
        
        # Setup mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Setup mock Pinecone
        mock_pinecone_instance = Mock()
        mock_pinecone_instance.as_retriever.return_value = Mock()
        mock_pinecone.return_value = mock_pinecone_instance
        
        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        # Setup mock hub prompts
        mock_hub.pull.return_value = Mock()
        
        # Setup mock chains
        mock_stuff_chain.return_value = Mock()
        mock_history_retriever.return_value = Mock()
        
        # Setup mock retrieval chain with chat history
        mock_retrieval_chain_instance = Mock()
        def mock_invoke(input):
            # Return response with the actual chat history that was passed
            return {
                "input": input,
                "answer": "This is a test answer",
                "context": MOCK_DOCS
            }
        mock_retrieval_chain_instance.invoke = mock_invoke
        mock_retrieval_chain.return_value = mock_retrieval_chain_instance
        
        yield {
            'embeddings': mock_embeddings,
            'pinecone': mock_pinecone,
            'llm': mock_llm,
            'hub': mock_hub,
            'stuff_chain': mock_stuff_chain,
            'history_retriever': mock_history_retriever,
            'retrieval_chain': mock_retrieval_chain
        }

def test_run_llm_basic_query(mock_dependencies):
    """Test basic query without chat history"""
    query = "test query"
    result = run_llm(query)
    
    assert isinstance(result, dict)
    assert 'query' in result
    assert 'result' in result
    assert 'source_document' in result
    assert result['result'] == "This is a test answer"
    assert result['source_document'] == MOCK_DOCS

def test_run_llm_with_chat_history(mock_dependencies):
    """Test query with chat history"""
    query = "test query"
    chat_history = [
        {"user": "previous question", "assistant": "previous answer"}
    ]
    result = run_llm(query, chat_history)
    
    assert isinstance(result, dict)
    assert 'query' in result
    assert 'result' in result
    assert 'source_document' in result
    assert result['result'] == "This is a test answer"
    assert result['source_document'] == MOCK_DOCS
    # Test that the function correctly processes the input query
    assert isinstance(result['query'], dict)
    assert 'input' in result['query']
    assert 'chat_history' in result['query']
    assert result['query']['input'] == query

def test_run_llm_dependency_initialization(mock_dependencies):
    """Test that all dependencies are properly initialized"""
    query = "test query"
    run_llm(query)
    
    # Verify embeddings initialization
    mock_dependencies['embeddings'].assert_called_once_with(model="embed-english-v3.0")
    
    # Verify Pinecone initialization
    mock_dependencies['pinecone'].assert_called_once()
    
    # Verify LLM initialization
    mock_dependencies['llm'].assert_called_once_with(
        model="gemini-2.0-flash-exp",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        top_p=0.95,
        top_k=40,
        max_tokens=2048,
        verbose=True
    )

def test_run_llm_error_handling():
    """Test error handling when dependencies fail"""
    with patch('core.CohereEmbeddings', side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            run_llm("test query")
        assert "API Error" in str(exc_info.value)
        
        