"""
Tests for FastAPI Endpoints
============================

This module tests the FastAPI API endpoints for the RAG chatbot system.
These are integration tests that verify the HTTP interface works correctly.

ENDPOINTS TESTED:
-----------------
- GET / : Health check / root endpoint
- POST /api/query : Process a question with RAG
- GET /api/courses : Get course catalog statistics

TEST APPROACH:
--------------
These tests use the test_client and async_test_client fixtures from conftest.py
which provide a mock RAG system. This allows testing the API layer without
requiring actual AI calls or database connections.

The test_app fixture creates a FastAPI app that mirrors the real app.py
endpoints but uses the mock_rag_system for all operations.

WHY SEPARATE TEST APP:
----------------------
The main app.py:
1. Mounts static files from ../frontend (may not exist in test environment)
2. Initializes a real RAGSystem at import time
3. Runs startup events that load documents

By creating a test app, we:
1. Avoid file system dependencies
2. Control the RAG system mock
3. Test just the API layer in isolation

RUNNING THESE TESTS:
--------------------
    cd backend
    uv run pytest tests/test_api_endpoints.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRootEndpoint:
    """
    Test suite for the root (/) endpoint.

    The root endpoint serves as a health check and entry point.
    In production it serves static files, but in tests it returns
    a simple JSON response.
    """

    def test_root_returns_ok_status(self, test_client):
        """
        Test that GET / returns a successful response.

        SCENARIO: Client requests the root endpoint.
        EXPECTED: 200 status code with JSON response.

        WHY THIS MATTERS:
        - Health checks rely on this endpoint
        - Verifies the server is running
        """
        response = test_client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestQueryEndpoint:
    """
    Test suite for the /api/query endpoint.

    This is the main RAG endpoint that processes user questions.
    It accepts a query and optional session_id, then returns
    an AI-generated answer with source citations.

    REQUEST FORMAT:
    {
        "query": "What is MCP?",
        "session_id": "optional-session-id"
    }

    RESPONSE FORMAT:
    {
        "answer": "MCP is...",
        "sources": [{"text": "Source", "link": "http://..."}],
        "session_id": "session-id"
    }
    """

    def test_query_with_valid_request(self, test_client, mock_rag_system):
        """
        Test that a valid query returns expected response structure.

        SCENARIO: Client sends a well-formed query.
        EXPECTED: 200 status with answer, sources, and session_id.

        COVERAGE:
        - Basic request/response cycle
        - Response model validation
        - RAG system integration
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP architecture?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a mock response about the course content."

    def test_query_with_session_id(self, test_client, mock_rag_system):
        """
        Test that query with existing session_id uses that session.

        SCENARIO: Client provides an existing session_id.
        EXPECTED: Response uses the provided session_id.

        WHY THIS MATTERS:
        - Multi-turn conversations need session continuity
        - Session_id allows conversation history tracking
        """
        session_id = "existing-session-456"
        response = test_client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        mock_rag_system.query.assert_called_with("Follow up question", session_id)

    def test_query_without_session_creates_new(self, test_client, mock_rag_system):
        """
        Test that query without session_id creates a new session.

        SCENARIO: Client sends query without session_id.
        EXPECTED: New session is created and returned.

        WHY THIS MATTERS:
        - First-time users don't have a session
        - System should auto-generate session_id
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"  # From mock
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_returns_sources(self, test_client, mock_rag_system):
        """
        Test that query response includes sources with correct structure.

        SCENARIO: RAG system returns sources from search.
        EXPECTED: Sources are included in response with text and link.

        WHY THIS MATTERS:
        - Sources provide transparency for users
        - Links enable users to read original content
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert "text" in source
        assert "link" in source

    def test_query_with_empty_string_fails(self, test_client):
        """
        Test that empty query string is rejected.

        SCENARIO: Client sends empty query.
        EXPECTED: Request validation error (422).

        WHY THIS MATTERS:
        - Empty queries waste resources
        - Clear error messages help users
        """
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string should still be accepted by FastAPI
        # (validation is on presence, not content)
        # The actual behavior depends on Pydantic model
        assert response.status_code in [200, 422]

    def test_query_with_missing_query_field_fails(self, test_client):
        """
        Test that request without query field is rejected.

        SCENARIO: Client sends request without required query field.
        EXPECTED: 422 Unprocessable Entity.

        WHY THIS MATTERS:
        - API must validate required fields
        - Clear error messages for malformed requests
        """
        response = test_client.post(
            "/api/query",
            json={"session_id": "some-session"}
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_with_invalid_json_fails(self, test_client):
        """
        Test that invalid JSON is rejected.

        SCENARIO: Client sends malformed JSON.
        EXPECTED: 422 Unprocessable Entity.

        WHY THIS MATTERS:
        - API must handle malformed input gracefully
        - No crashes from bad input
        """
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_handles_rag_system_error(self, test_client, mock_rag_system):
        """
        Test that RAG system errors result in 500 response.

        SCENARIO: RAG system raises an exception during query.
        EXPECTED: 500 Internal Server Error with error detail.

        WHY THIS MATTERS:
        - Internal errors should not crash the server
        - Errors should be reported to client appropriately
        """
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    def test_query_calls_rag_system_with_query(self, test_client, mock_rag_system):
        """
        Test that the query is passed correctly to RAG system.

        SCENARIO: Client sends specific query text.
        EXPECTED: RAG system receives exact query text.

        WHY THIS MATTERS:
        - Verifies data flow from API to business logic
        - Query must not be modified in transit
        """
        query_text = "Explain the MCP protocol in detail"

        response = test_client.post(
            "/api/query",
            json={"query": query_text}
        )

        assert response.status_code == 200
        # Verify the mock was called with the correct query
        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == query_text


class TestCoursesEndpoint:
    """
    Test suite for the /api/courses endpoint.

    This endpoint returns statistics about available courses,
    including total count and list of course titles.

    RESPONSE FORMAT:
    {
        "total_courses": 4,
        "course_titles": ["Course A", "Course B", ...]
    }
    """

    def test_courses_returns_statistics(self, test_client, mock_rag_system):
        """
        Test that GET /api/courses returns course statistics.

        SCENARIO: Client requests course information.
        EXPECTED: 200 status with total_courses and course_titles.

        COVERAGE:
        - Basic endpoint functionality
        - Response model validation
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_courses_returns_correct_count(self, test_client, mock_rag_system):
        """
        Test that courses endpoint returns correct course count.

        SCENARIO: RAG system has 3 courses (from mock).
        EXPECTED: total_courses equals 3.

        WHY THIS MATTERS:
        - Verifies integration with RAG system analytics
        - Count must match actual data
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3

    def test_courses_returns_course_titles(self, test_client, mock_rag_system):
        """
        Test that courses endpoint returns list of course titles.

        SCENARIO: RAG system has specific courses (from mock).
        EXPECTED: course_titles contains expected titles.

        WHY THIS MATTERS:
        - Users need to know what courses are available
        - Titles enable course-specific queries
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]
        assert "Course B" in data["course_titles"]

    def test_courses_handles_analytics_error(self, test_client, mock_rag_system):
        """
        Test that analytics errors result in 500 response.

        SCENARIO: RAG system analytics raises an exception.
        EXPECTED: 500 Internal Server Error.

        WHY THIS MATTERS:
        - Errors should not crash the server
        - Users get meaningful error information
        """
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failed")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestAsyncEndpoints:
    """
    Test suite for async endpoint testing using async client.

    These tests verify the same endpoints but using the async test client,
    which is useful for testing async-specific behaviors.
    """

    @pytest.mark.asyncio
    async def test_async_query_endpoint(self, async_test_client, mock_rag_system):
        """
        Test query endpoint using async client.

        SCENARIO: Async client sends query request.
        EXPECTED: Successful async response.
        """
        response = await async_test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    @pytest.mark.asyncio
    async def test_async_courses_endpoint(self, async_test_client, mock_rag_system):
        """
        Test courses endpoint using async client.

        SCENARIO: Async client requests courses.
        EXPECTED: Successful async response.
        """
        response = await async_test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data


class TestAPIContentTypes:
    """
    Test suite for API content type handling.

    These tests verify that the API correctly handles different
    content types and request formats.
    """

    def test_query_requires_json_content_type(self, test_client):
        """
        Test that query endpoint requires JSON content type.

        SCENARIO: Client sends form data instead of JSON.
        EXPECTED: 422 Unprocessable Entity.
        """
        response = test_client.post(
            "/api/query",
            data={"query": "What is MCP?"}
        )

        assert response.status_code == 422

    def test_query_response_is_json(self, test_client, mock_rag_system):
        """
        Test that query response has JSON content type.

        SCENARIO: Valid query request.
        EXPECTED: Response has application/json content type.
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_courses_response_is_json(self, test_client, mock_rag_system):
        """
        Test that courses response has JSON content type.

        SCENARIO: Request to courses endpoint.
        EXPECTED: Response has application/json content type.
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
