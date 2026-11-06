from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from rag_evals.pipeline import RAGPipeline, RAGResponse
from rag_evals.generator import GENERATOR_SYSTEM_PROMPT, MODEL_ID


@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="Regression testing ensures previous code remains functional.",
            metadata={"session": "8", "chunk_id": 1},
        ),
        Document(
            page_content="DeepEval can run automated evaluation suites on regressions.",
            metadata={"session": "8", "chunk_id": 2},
        ),
        Document(
            page_content="Baselines serve as the reference point for comparative evals.",
            metadata={"session": "3", "chunk_id": 3},
        ),
    ]


@pytest.fixture
def mock_retriever(mock_documents):
    retriever = MagicMock()
    retriever.invoke.return_value = mock_documents
    return retriever


@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Regression testing is running evals to prevent regressions."
    client.models.generate_content.return_value = mock_response
    return client


def test_pipeline_initialization_defaults(mock_retriever, mock_genai_client):
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
    )
    assert pipeline.model == MODEL_ID
    assert pipeline.top_k == 5
    assert pipeline.system_prompt == GENERATOR_SYSTEM_PROMPT
    assert pipeline.retriever == mock_retriever
    assert pipeline.client == mock_genai_client


def test_pipeline_custom_model_and_prompt(mock_retriever, mock_genai_client):
    custom_prompt = "Custom system instructions."
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
        model="gemini-3.5-flash",
        top_k=3,
        system_prompt=custom_prompt,
    )
    assert pipeline.model == "gemini-3.5-flash"
    assert pipeline.top_k == 3
    assert pipeline.system_prompt == custom_prompt


def test_retrieve_slices_top_k(mock_retriever, mock_genai_client, mock_documents):
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
        top_k=5,
    )
    docs = pipeline.retrieve("test query", top_k=2)
    assert len(docs) == 2
    assert docs[0].page_content == mock_documents[0].page_content
    assert docs[1].page_content == mock_documents[1].page_content


def test_generate_formats_prompt_and_calls_sdk(mock_retriever, mock_genai_client):
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
        model="gemini-3.8-flash",
    )
    chunks = ["Fact A is true.", "Fact B is also true."]
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        answer = pipeline.generate("What is Fact A?", chunks)

    assert answer == "Regression testing is running evals to prevent regressions."
    mock_genai_client.models.generate_content.assert_called_once()
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert "[1] Fact A is true." in call_kwargs["contents"]
    assert "[2] Fact B is also true." in call_kwargs["contents"]
    assert "Question: What is Fact A?" in call_kwargs["contents"]
    assert call_kwargs["config"].temperature == 0.0
    assert call_kwargs["config"].system_instruction == GENERATOR_SYSTEM_PROMPT


def test_query_end_to_end_mocked(mock_retriever, mock_genai_client, mock_documents):
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
        model="gemini-3.8-flash",
        top_k=2,
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        response = pipeline.query("What is regression testing?")

    assert isinstance(response, RAGResponse)
    assert response.query == "What is regression testing?"
    assert response.answer == "Regression testing is running evals to prevent regressions."
    assert len(response.context_chunks) == 2
    assert response.context_chunks[0] == mock_documents[0].page_content
    assert response.context_chunks[1] == mock_documents[1].page_content
    assert len(response.context_documents) == 2
    assert response.context_documents[0]["metadata"]["session"] == "8"
    assert response.latency_seconds >= 0.0
    assert response.model == "gemini-3.8-flash"


def test_negative_constraint_abstention(mock_retriever, mock_genai_client):
    refusal_msg = "I cannot answer this question based on the provided context."
    mock_genai_client.models.generate_content.return_value.text = refusal_msg

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        response = pipeline.run("Unrelated query")

    assert response.answer == refusal_msg


def test_batch_query_mocked(mock_retriever, mock_genai_client):
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        client=mock_genai_client,
    )
    queries = ["Question 1", "Question 2"]
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        with patch("time.sleep") as mock_sleep:
            results = pipeline.batch_query(queries, delay_seconds=5.0)

    assert len(results) == 2
    assert results[0].query == "Question 1"
    assert results[1].query == "Question 2"
    # Delay should only be called once between question 1 and question 2
    mock_sleep.assert_called_once_with(5.0)
