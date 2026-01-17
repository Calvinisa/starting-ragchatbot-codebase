"""
Shared Test Fixtures for RAG System Tests
==========================================

This module contains pytest fixtures that are automatically available to all test
files in the tests directory. Fixtures provide reusable test setup and mock objects
that simulate real components without requiring actual API calls or database connections.

WHY USE FIXTURES?
-----------------
Fixtures help us:
1. Avoid code duplication across test files
2. Ensure consistent test data across all tests
3. Isolate tests from external dependencies (APIs, databases)
4. Make tests fast and reliable (no network calls)

HOW PYTEST FIXTURES WORK:
-------------------------
When a test function has a parameter matching a fixture name, pytest automatically
calls the fixture and passes its return value to the test. For example:

    def test_something(mock_vector_store):  # pytest provides mock_vector_store
        mock_vector_store.search(...)

Fixtures defined here are available to ALL test files in this directory.

FIXTURE CATEGORIES:
-------------------
1. Configuration Fixtures: MockConfig, mock_config
2. Component Fixtures: mock_vector_store, mock_rag_system
3. Data Fixtures: sample_search_results, empty_search_results, error_search_results
4. API Testing Fixtures: test_client, async_test_client
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Generator

# Add backend to path so we can import our application modules
# This is necessary because tests run from a different directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


@dataclass
class MockConfig:
    """
    Mock configuration object that mirrors the real Config class.

    This provides test values for all configuration parameters without requiring
    a .env file or real API keys. All values are safe defaults suitable for testing.

    Attributes:
        ANTHROPIC_API_KEY: Fake API key (never used in actual API calls during tests)
        ANTHROPIC_MODEL: Model identifier (tests mock the API, so this isn't called)
        EMBEDDING_MODEL: The sentence transformer model name for embeddings
        CHUNK_SIZE: Number of characters per document chunk
        CHUNK_OVERLAP: Overlap between consecutive chunks for context continuity
        MAX_RESULTS: Maximum number of search results to return
        MAX_HISTORY: Number of conversation turns to retain in session
        CHROMA_PATH: Path for ChromaDB storage (uses test-specific directory)

    Usage:
        config = MockConfig()
        rag_system = RAGSystem(config)
    """

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """
    Provide a mock configuration object for tests that need config values.

    Returns:
        MockConfig: A dataclass instance with test-safe configuration values.

    Example:
        def test_system_init(mock_config):
            system = RAGSystem(mock_config)
            assert system.config.MAX_RESULTS == 5
    """
    return MockConfig()


@pytest.fixture
def mock_vector_store():
    """
    Create a mock VectorStore object for testing components that depend on it.

    This fixture creates a Mock object that simulates the VectorStore class without
    requiring an actual ChromaDB database or embedding model. Tests can configure
    the mock's return values to simulate different scenarios.

    The mock includes:
    - course_catalog: Mock ChromaDB collection for course metadata
    - course_content: Mock ChromaDB collection for searchable content

    Returns:
        Mock: A mock VectorStore with pre-configured collection attributes.

    Example:
        def test_search(mock_vector_store, sample_search_results):
            # Configure what the mock should return
            mock_vector_store.search.return_value = sample_search_results

            # Now use the mock in your test
            tool = CourseSearchTool(mock_vector_store)
            result = tool.execute(query="test")

            # Verify the mock was called correctly
            mock_vector_store.search.assert_called_once()

    Note:
        You must configure mock_vector_store.search.return_value before calling
        any method that uses search. Use sample_search_results, empty_search_results,
        or error_search_results fixtures for common scenarios.
    """
    mock_store = Mock()

    # Mock the two ChromaDB collections that VectorStore manages
    # course_catalog: stores course titles, instructors, and lesson lists
    # course_content: stores searchable text chunks with metadata
    mock_store.course_catalog = Mock()
    mock_store.course_content = Mock()

    return mock_store


@pytest.fixture
def sample_search_results():
    """
    Provide realistic search results for testing successful search scenarios.

    This fixture simulates what VectorStore.search() returns when it finds
    relevant content. It includes:
    - Two document chunks with MCP-related content
    - Metadata with course title, lesson number, and chunk index
    - Distance scores (lower = more relevant, 0.1 is very relevant)
    - No error condition

    Returns:
        SearchResults: A dataclass with documents, metadata, distances, and no error.

    Use Cases:
    - Testing that search results are properly formatted for display
    - Testing that tool execution correctly processes found content
    - Testing that sources are correctly extracted from results

    Example:
        def test_format_results(mock_vector_store, sample_search_results):
            mock_vector_store.search.return_value = sample_search_results
            tool = CourseSearchTool(mock_vector_store)
            result = tool.execute(query="MCP")
            assert "Lesson" in result  # Results should include lesson info
    """
    return SearchResults(
        documents=[
            "This is content about MCP architecture and how it works.",
            "More content about building MCP servers.",
        ],
        metadata=[
            {
                "course_title": "Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 2,
                "chunk_index": 0,
            },
            {
                "course_title": "Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 4,
                "chunk_index": 0,
            },
        ],
        distances=[0.1, 0.2],  # Lower distance = more semantically similar
        error=None,  # No error means search succeeded
    )


@pytest.fixture
def empty_search_results():
    """
    Provide empty search results for testing "no results found" scenarios.

    This fixture simulates what VectorStore.search() returns when the query
    doesn't match any content in the database. This is different from an error -
    the search executed successfully but found nothing relevant.

    Returns:
        SearchResults: A dataclass with empty lists and no error.

    Use Cases:
    - Testing graceful handling when no content matches the query
    - Testing appropriate user messaging for empty results
    - Testing that the system doesn't crash on empty data

    Example:
        def test_no_results_message(mock_vector_store, empty_search_results):
            mock_vector_store.search.return_value = empty_search_results
            tool = CourseSearchTool(mock_vector_store)
            result = tool.execute(query="xyz123")
            assert "No relevant content found" in result
    """
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None,  # No error - search worked, just found nothing
    )


@pytest.fixture
def error_search_results():
    """
    Provide search results with an error for testing error handling scenarios.

    This fixture simulates what VectorStore.search() returns when an error occurs
    during the search process, such as:
    - Course name doesn't match any known course (fuzzy matching failed)
    - Database connection issues
    - Invalid query parameters

    Returns:
        SearchResults: A dataclass with empty lists and an error message.

    Use Cases:
    - Testing that error messages are properly propagated to the user
    - Testing that the system gracefully handles search failures
    - Testing error message formatting in responses

    Example:
        def test_error_handling(mock_vector_store, error_search_results):
            mock_vector_store.search.return_value = error_search_results
            tool = CourseSearchTool(mock_vector_store)
            result = tool.execute(query="test", course_name="nonexistent")
            assert "No course found" in result
    """
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'nonexistent'",  # Error from fuzzy course matching
    )


# =============================================================================
# API TESTING FIXTURES
# =============================================================================

@pytest.fixture
def mock_rag_system(sample_search_results):
    """
    Create a mock RAGSystem for API testing.

    This fixture creates a mock RAGSystem that can be used to test API endpoints
    without requiring actual AI calls or database connections. The mock is
    pre-configured with reasonable defaults that can be overridden per test.

    Returns:
        Mock: A mock RAGSystem with pre-configured return values.

    Pre-configured Behaviors:
        - query(): Returns ("Mock response", [{"text": "Source 1", "link": "http://example.com"}])
        - get_course_analytics(): Returns {"total_courses": 3, "course_titles": ["Course A", "Course B", "Course C"]}
        - session_manager.create_session(): Returns "test-session-123"

    Example:
        def test_api_query(mock_rag_system, test_client):
            mock_rag_system.query.return_value = ("Custom response", [])
            response = test_client.post("/api/query", json={"query": "test"})
            assert response.json()["answer"] == "Custom response"
    """
    mock_system = Mock()

    # Configure query method
    mock_system.query.return_value = (
        "This is a mock response about the course content.",
        [{"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"}]
    )

    # Configure analytics method
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }

    # Configure session manager
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-123"

    return mock_system


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a FastAPI test app without static file mounting.

    The main app.py mounts static files that may not exist in the test environment.
    This fixture creates a minimal FastAPI app with just the API endpoints for testing.

    Returns:
        FastAPI: A test application with API endpoints but no static file handling.

    Why Not Import app.py Directly:
        - app.py mounts static files from ../frontend which may not exist
        - app.py initializes a real RAGSystem at import time
        - Tests need to control the RAGSystem mock

    Example:
        def test_endpoint(test_app):
            client = TestClient(test_app)
            response = client.get("/api/courses")
            assert response.status_code == 200
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test")

    # Pydantic models (mirrors app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints (mirrors app.py)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Health check endpoint for testing."""
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """
    Create a synchronous test client for API testing.

    Uses Starlette's TestClient which is the standard approach for testing
    FastAPI/Starlette applications synchronously.

    Returns:
        TestClient: A synchronous HTTP client configured for the test app.

    Example:
        def test_courses_endpoint(test_client):
            response = test_client.get("/api/courses")
            assert response.status_code == 200
            assert "total_courses" in response.json()
    """
    from starlette.testclient import TestClient

    with TestClient(test_app) as client:
        yield client


@pytest.fixture
async def async_test_client(test_app):
    """
    Create an async test client for API testing.

    Uses httpx.AsyncClient for testing async endpoints. Use this when you
    need to test async behaviors or when working with async fixtures.

    Returns:
        httpx.AsyncClient: An async HTTP client configured for the test app.

    Example:
        async def test_async_query(async_test_client):
            response = await async_test_client.post(
                "/api/query",
                json={"query": "What is MCP?"}
            )
            assert response.status_code == 200
    """
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
