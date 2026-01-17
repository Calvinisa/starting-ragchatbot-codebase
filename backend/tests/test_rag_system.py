"""
Tests for RAG System Content-Query Handling
=============================================

This module tests the RAGSystem class - the main orchestrator that brings together
all components (VectorStore, AIGenerator, ToolManager, SessionManager) to handle
user queries about course content.

WHAT IS RAGSystem?
------------------
RAGSystem is the central orchestrator of the RAG chatbot. It:
1. Initializes all components (vector store, AI generator, tools)
2. Handles user queries through the query() method
3. Manages conversation sessions
4. Coordinates tool execution between Claude and the vector store
5. Returns responses with source citations

ARCHITECTURE OVERVIEW:
----------------------
    User Query
         |
         v
    RAGSystem.query()
         |
         +---> SessionManager (get conversation history)
         |
         +---> AIGenerator.generate_response()
         |         |
         |         +---> Claude API (with tools)
         |         |
         |         +---> ToolManager.execute_tool() (if tool_use)
         |                    |
         |                    +---> CourseSearchTool.execute()
         |                              |
         |                              +---> VectorStore.search()
         |
         v
    (response, sources)

TEST COVERAGE OVERVIEW:
-----------------------
TestRAGSystemContentQueries (6 tests):
    - Tool definitions are passed to AI generator
    - Content questions trigger the agentic loop
    - Sources are returned after search
    - Sources are reset between queries
    - Search errors are handled gracefully
    - Empty search results are handled

TestRAGSystemToolRegistration (3 tests):
    - CourseSearchTool is registered
    - CourseOutlineTool is registered
    - Both tools are available

TestVectorStoreSearch (2 tests):
    - VectorStore.search() calls ChromaDB
    - Search returns SearchResults dataclass

MOCKING STRATEGY:
-----------------
RAGSystem has many dependencies. These tests mock:
- VectorStore: Avoids ChromaDB and embedding model initialization
- DocumentProcessor: Avoids file system access
- SessionManager: Avoids session state complexity
- Anthropic client: Avoids API calls

This isolation ensures tests are fast, reliable, and don't require
external services or API keys.

FIXTURES:
---------
TestRAGSystemContentQueries uses:
    - mock_dependencies: Dictionary of all mocked components
    - rag_system: Fully initialized RAGSystem with mocks

TestRAGSystemToolRegistration uses:
    - rag_system: Minimal RAGSystem for tool registration testing

TestVectorStoreSearch uses:
    - mock_chroma_client: Mock ChromaDB client

RUNNING THESE TESTS:
--------------------
    cd backend
    uv run pytest tests/test_rag_system.py -v
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from search_tools import ToolManager, CourseSearchTool


@dataclass
class MockConfig:
    """
    Mock configuration for RAGSystem testing.

    Mirrors the real Config class but with test-safe values.
    No real API keys or file paths are used.

    Note: This is duplicated from conftest.py because these tests
    need it locally for the mock_dependencies fixture. In a larger
    codebase, you might refactor to share this.
    """
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@dataclass
class MockToolUseBlock:
    """
    Mock for Anthropic tool_use content block.

    Simulates Claude's response when it decides to use a tool.
    See test_ai_generator.py for detailed documentation.

    Note: Duplicated here because these tests define their own
    response sequences independent of AIGenerator tests.
    """
    type: str = "tool_use"
    id: str = "tool_123"
    name: str = "search_course_content"
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {"query": "MCP architecture"}


@dataclass
class MockTextBlock:
    """
    Mock for Anthropic text content block.

    Simulates Claude's final text response.
    See test_ai_generator.py for detailed documentation.
    """
    type: str = "text"
    text: str = "Here is the answer."


class TestRAGSystemContentQueries:
    """
    Test suite for RAG system handling of content-related queries.

    These tests verify the complete flow from user query to response,
    including tool execution and source tracking. This is integration-level
    testing of the RAGSystem class.

    KEY BEHAVIORS TESTED:
    1. RAGSystem correctly wires together all components
    2. User queries trigger the appropriate tool execution
    3. Sources are properly tracked and returned
    4. Errors in any component are handled gracefully

    INTEGRATION FOCUS:
    While individual components are tested elsewhere, these tests verify
    that RAGSystem correctly orchestrates them together.
    """

    @pytest.fixture
    def mock_dependencies(self):
        """
        Set up all mock dependencies for RAGSystem.

        This fixture creates a dictionary of mocks for all RAGSystem dependencies,
        pre-configured with sensible defaults. Tests can override specific behaviors.

        Returns:
            dict: Contains 'vector_store' and 'anthropic_client' mocks

        Mock Configurations:
        - vector_store.search: Returns sample search results
        - vector_store.get_lesson_link: Returns example URL
        - vector_store.get_course_count: Returns 4 courses
        - vector_store.get_existing_course_titles: Returns test titles
        - anthropic_client: Ready to be configured per-test

        Usage:
            def test_something(self, mock_dependencies):
                # Override search to return empty results
                mock_dependencies['vector_store'].search.return_value = SearchResults(...)
        """
        mocks = {}

        # Mock VectorStore with complete interface
        mocks['vector_store'] = Mock()
        mocks['vector_store'].search.return_value = SearchResults(
            documents=["Content about MCP"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
            error=None
        )
        mocks['vector_store'].get_lesson_link.return_value = "https://example.com/lesson"
        mocks['vector_store'].get_course_count.return_value = 4
        mocks['vector_store'].get_existing_course_titles.return_value = ["MCP Course"]
        mocks['vector_store'].course_catalog = Mock()
        mocks['vector_store'].course_content = Mock()

        # Mock Anthropic client (will be configured per-test)
        mocks['anthropic_client'] = Mock()

        return mocks

    @pytest.fixture
    def rag_system(self, mock_dependencies):
        """
        Create RAGSystem with all dependencies mocked.

        This fixture patches all external dependencies so RAGSystem can be
        instantiated without:
        - ChromaDB database
        - Sentence transformer model
        - Document files
        - Anthropic API key

        Returns:
            RAGSystem: Fully initialized system with mocked dependencies

        Note:
            After patching, we also directly assign the mocks to ensure
            they're properly connected (patches affect initialization,
            but we need references for assertions).
        """
        with patch('rag_system.VectorStore', return_value=mock_dependencies['vector_store']), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'), \
             patch('ai_generator.anthropic.Anthropic', return_value=mock_dependencies['anthropic_client']):

            config = MockConfig()
            system = RAGSystem(config)
            # Ensure mocks are connected for assertion purposes
            system.vector_store = mock_dependencies['vector_store']
            system.ai_generator.client = mock_dependencies['anthropic_client']
            return system

    def test_query_passes_tools_to_ai_generator(self, rag_system, mock_dependencies):
        """
        Test that query method provides tools to ai_generator.

        SCENARIO: User asks a question through RAGSystem.query().
        EXPECTED: The Claude API call includes tool definitions.

        WHY THIS MATTERS:
        - Tools must be passed for Claude to use them
        - This verifies RAGSystem's wiring between ToolManager and AIGenerator
        - Without this, no RAG functionality would work

        COVERAGE:
        - Tool definitions are extracted from ToolManager
        - Tools are passed to Claude API call
        - search_course_content tool is included

        INTEGRATION POINT:
        RAGSystem.query() -> AIGenerator.generate_response(tools=...)
        """
        # Arrange: Set up simple response (no tool use)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MockTextBlock()]
        mock_dependencies['anthropic_client'].messages.create.return_value = mock_response

        # Act
        response, sources = rag_system.query("What is MCP?")

        # Assert: Tools were passed to API call
        call_args = mock_dependencies['anthropic_client'].messages.create.call_args
        assert "tools" in call_args.kwargs
        tools = call_args.kwargs["tools"]
        tool_names = [t["name"] for t in tools]
        assert "search_course_content" in tool_names

    def test_query_handles_tool_execution_for_content_questions(self, rag_system, mock_dependencies):
        """
        Test that content questions trigger the complete tool execution loop.

        SCENARIO: User asks about course content. Claude decides to search,
                  executes the search, and returns a response.
        EXPECTED: Two API calls (tool_use then end_turn), search is executed.

        WHY THIS MATTERS:
        - This is the core RAG flow
        - Verifies the complete agentic loop works through RAGSystem
        - Ensures all components are properly connected

        COVERAGE:
        - Full agentic loop (tool_use -> execute -> result -> response)
        - VectorStore.search is called
        - Final response contains expected content

        FLOW TESTED:
        query() -> generate_response() -> [tool_use] -> execute_tool() ->
        search() -> [tool_result] -> generate_response() -> [text] -> return
        """
        # Arrange: Set up tool_use then text response sequence
        tool_use_block = MockToolUseBlock(
            input={"query": "MCP architecture"}
        )

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="MCP architecture allows...")]

        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]

        # Act
        response, sources = rag_system.query("Tell me about MCP architecture")

        # Assert: Complete loop executed
        assert mock_dependencies['anthropic_client'].messages.create.call_count == 2
        assert "MCP architecture" in response
        mock_dependencies['vector_store'].search.assert_called_once()

    def test_query_returns_sources_after_search(self, rag_system, mock_dependencies):
        """
        Test that query returns sources from search tool for citations.

        SCENARIO: After a successful search and response, the UI needs
                  to display source citations.
        EXPECTED: query() returns (response, sources) with populated sources.

        WHY THIS MATTERS:
        - Source citations are crucial for transparency
        - Users need to know where information comes from
        - This enables the "Sources" section in the UI

        COVERAGE:
        - Sources are collected from tool execution
        - Sources are returned as second element of tuple
        - Sources contain expected structure

        UI INTEGRATION:
        The sources returned here are displayed as clickable links
        in the chat interface.
        """
        # Arrange: Set up tool execution sequence
        tool_use_block = MockToolUseBlock()

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock()]

        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]

        # Act
        response, sources = rag_system.query("What is MCP?")

        # Assert: Sources are returned
        assert len(sources) > 0
        assert any("text" in source for source in sources)

    def test_query_resets_sources_after_retrieval(self, rag_system, mock_dependencies):
        """
        Test that sources are reset after being retrieved (no carryover).

        SCENARIO: User asks two questions in sequence. Sources from the first
                  query should not appear in the second query's results.
        EXPECTED: After each query, sources are fresh (not accumulated).

        WHY THIS MATTERS:
        - Sources must match the current query only
        - Stale sources would be confusing and incorrect
        - This verifies proper state management

        COVERAGE:
        - Source state is reset between queries
        - get_last_sources() returns empty after retrieval
        - No source leakage between queries

        BUG PREVENTION:
        Without proper reset, sources would accumulate, showing
        irrelevant citations from previous questions.
        """
        # Arrange: Set up tool execution sequence
        tool_use_block = MockToolUseBlock()

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock()]

        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]

        # Act: First query
        response1, sources1 = rag_system.query("First query")

        # Reset mocks for second query
        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]
        mock_dependencies['vector_store'].search.reset_mock()

        # Act: Second query
        response2, sources2 = rag_system.query("Second query")

        # Assert: Sources are reset after retrieval
        assert rag_system.tool_manager.get_last_sources() == []

    def test_query_handles_search_error_gracefully(self, rag_system, mock_dependencies):
        """
        Test that query handles search errors without crashing.

        SCENARIO: User asks about a course that doesn't exist. VectorStore
                  returns an error, but the system should still respond.
        EXPECTED: No exception is raised, response is returned.

        WHY THIS MATTERS:
        - Errors in search shouldn't crash the entire system
        - Claude can still provide a helpful response even with errors
        - User experience is preserved

        COVERAGE:
        - Error propagation through tool execution
        - Graceful degradation of functionality
        - System stability under error conditions

        ERROR HANDLING FLOW:
        VectorStore returns error -> Tool returns error message ->
        Claude receives error -> Claude explains to user
        """
        # Arrange: Configure search to return error
        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'nonexistent'"
        )

        tool_use_block = MockToolUseBlock(
            input={"query": "test", "course_name": "nonexistent"}
        )

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="I could not find information about that.")]

        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]

        # Act: Should not raise exception
        response, sources = rag_system.query("Tell me about nonexistent course")

        # Assert: Response is returned, no crash
        assert response is not None

    def test_query_handles_empty_search_results(self, rag_system, mock_dependencies):
        """
        Test that query handles empty search results gracefully.

        SCENARIO: User asks a valid question but no matching content exists.
                  Search returns empty (not an error, just no matches).
        EXPECTED: System responds appropriately without crashing.

        WHY THIS MATTERS:
        - Empty results != error (important distinction)
        - Claude should explain that no content was found
        - System should remain stable

        COVERAGE:
        - Empty results handling (different from error)
        - Appropriate response generation
        - No false positives or crashes

        EDGE CASE:
        This tests the boundary between "nothing found" and "error occurred".
        """
        # Arrange: Configure search to return empty results (no error)
        mock_dependencies['vector_store'].search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None  # No error, just empty
        )

        tool_use_block = MockToolUseBlock()

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="No relevant content found.")]

        mock_dependencies['anthropic_client'].messages.create.side_effect = [
            first_response, second_response
        ]

        # Act
        response, sources = rag_system.query("Very specific query")

        # Assert: Response is returned
        assert response is not None


class TestRAGSystemToolRegistration:
    """
    Test that RAG system correctly registers all tools at initialization.

    These tests verify that RAGSystem's __init__ properly sets up the
    ToolManager with all required tools. This is critical because if
    tools aren't registered, Claude can't use them.

    TOOLS EXPECTED:
    1. search_course_content: For searching course content
    2. get_course_outline: For getting course structure/outline
    """

    @pytest.fixture
    def rag_system(self):
        """
        Create RAGSystem with minimal mocking for tool registration testing.

        This fixture only needs to verify tool registration, so we use
        basic mocks without configuring complex behaviors.

        Returns:
            RAGSystem: Instance with tools registered
        """
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'), \
             patch('ai_generator.anthropic.Anthropic'):

            config = MockConfig()
            return RAGSystem(config)

    def test_search_tool_is_registered(self, rag_system):
        """
        Test that CourseSearchTool is registered in ToolManager.

        SCENARIO: RAGSystem is initialized.
        EXPECTED: ToolManager contains search_course_content tool.

        WHY THIS MATTERS:
        - Without this tool, Claude can't search course content
        - This is the primary RAG functionality
        - Verifies initialization wiring is correct
        """
        tool_names = [t["name"] for t in rag_system.tool_manager.get_tool_definitions()]
        assert "search_course_content" in tool_names

    def test_outline_tool_is_registered(self, rag_system):
        """
        Test that CourseOutlineTool is registered in ToolManager.

        SCENARIO: RAGSystem is initialized.
        EXPECTED: ToolManager contains get_course_outline tool.

        WHY THIS MATTERS:
        - Without this tool, Claude can't provide course outlines
        - Users asking "what topics are covered" need this
        - Verifies initialization wiring is correct
        """
        tool_names = [t["name"] for t in rag_system.tool_manager.get_tool_definitions()]
        assert "get_course_outline" in tool_names

    def test_both_tools_are_available(self, rag_system):
        """
        Test that both tools are registered and available.

        SCENARIO: RAGSystem is initialized.
        EXPECTED: Exactly 2 tools are registered.

        WHY THIS MATTERS:
        - Verifies complete tool registration
        - Catches accidental tool removal during refactoring
        - Documents expected number of tools

        NOTE:
        If you add more tools, update this test's assertion.
        """
        definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(definitions) == 2


class TestVectorStoreSearch:
    """
    Direct tests for VectorStore search functionality.

    These tests verify VectorStore's interaction with ChromaDB without
    going through the full RAG pipeline. This is useful for:
    1. Isolating VectorStore behavior
    2. Testing ChromaDB integration specifics
    3. Verifying return type contracts

    COMPONENT FOCUS:
    While other tests use mock VectorStore, these tests verify
    the actual VectorStore class with a mocked ChromaDB client.
    """

    @pytest.fixture
    def mock_chroma_client(self):
        """
        Create mock ChromaDB client with pre-configured query results.

        This simulates ChromaDB's response format without requiring
        an actual database. The format matches ChromaDB's query API.

        Returns:
            Mock: ChromaDB PersistentClient mock

        ChromaDB Query Format:
        {
            'documents': [[doc1, doc2, ...]],  # Nested list
            'metadatas': [[meta1, meta2, ...]],
            'distances': [[dist1, dist2, ...]]
        }
        """
        mock_client = Mock()
        mock_collection = Mock()

        # Mock query results in ChromaDB's format
        mock_collection.query.return_value = {
            'documents': [['Content about topic']],  # Nested list!
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }

        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client

    def test_search_calls_chroma_query(self, mock_chroma_client):
        """
        Test that VectorStore.search() calls ChromaDB query method.

        SCENARIO: VectorStore.search() is called with a query.
        EXPECTED: ChromaDB collection.query() is invoked.

        WHY THIS MATTERS:
        - Verifies VectorStore correctly delegates to ChromaDB
        - Ensures the search pipeline is connected
        - Basic smoke test for search functionality

        COVERAGE:
        - VectorStore initializes ChromaDB client
        - search() calls collection.query()
        """
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            from vector_store import VectorStore
            store = VectorStore("./test_db", "all-MiniLM-L6-v2")

            # Act
            results = store.search(query="test query")

            # Assert: ChromaDB query was called
            store.course_content.query.assert_called_once()

    def test_search_returns_search_results_object(self, mock_chroma_client):
        """
        Test that search returns SearchResults dataclass (not raw dict).

        SCENARIO: VectorStore.search() is called.
        EXPECTED: Returns SearchResults dataclass with expected attributes.

        WHY THIS MATTERS:
        - SearchResults provides type safety
        - Consistent interface for consumers
        - Easier error handling (error attribute)

        COVERAGE:
        - Return type is SearchResults
        - Required attributes exist (documents, metadata, error)

        TYPE CONTRACT:
        SearchResults has:
        - documents: List[str]
        - metadata: List[dict]
        - distances: List[float]
        - error: Optional[str]
        """
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            from vector_store import VectorStore, SearchResults
            store = VectorStore("./test_db", "all-MiniLM-L6-v2")

            # Act
            results = store.search(query="test query")

            # Assert: Correct return type with expected attributes
            assert isinstance(results, SearchResults)
            assert hasattr(results, 'documents')
            assert hasattr(results, 'metadata')
            assert hasattr(results, 'error')
