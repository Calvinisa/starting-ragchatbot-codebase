"""
Tests for CourseSearchTool and ToolManager
============================================

This module tests the search tool components that enable Claude to search course content.
These tests verify the tool's ability to query the vector store and format results for
the AI to use in generating responses.

WHAT IS CourseSearchTool?
-------------------------
CourseSearchTool is an Anthropic "tool" that Claude can call during a conversation.
When a user asks about course content, Claude decides to call this tool with search
parameters, the tool queries the vector database, and returns formatted results that
Claude uses to answer the question.

WHAT IS ToolManager?
--------------------
ToolManager is a registry that holds all available tools and routes tool execution
requests to the appropriate tool. It provides:
1. Tool registration (adding tools to the system)
2. Tool execution (calling the right tool with arguments)
3. Tool definition export (for sending to Claude API)

TEST COVERAGE OVERVIEW:
-----------------------
TestCourseSearchToolExecute (9 tests):
    - Basic search functionality and result formatting
    - Filter parameter passing (course_name, lesson_number)
    - Error handling (search errors, empty results)
    - Source tracking for citation purposes
    - Tool definition schema validation

TestToolManager (4 tests):
    - Tool registration
    - Tool execution routing
    - Unknown tool handling
    - Tool definition export

FIXTURES USED (from conftest.py):
---------------------------------
    - mock_vector_store: Mock VectorStore for simulating searches
    - sample_search_results: Realistic search results with content
    - empty_search_results: Empty results for "nothing found" scenarios
    - error_search_results: Results with error message

RUNNING THESE TESTS:
--------------------
    cd backend
    uv run pytest tests/test_course_search_tool.py -v
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """
    Test suite for CourseSearchTool.execute method.

    The execute() method is the core of the search tool - it's what gets called
    when Claude decides to search for course content. These tests verify that:

    1. Search results are properly formatted for Claude to understand
    2. Filter parameters are correctly passed to the vector store
    3. Errors are handled gracefully and communicated clearly
    4. Source citations are tracked for transparency

    Test Pattern Used: Arrange-Act-Assert (AAA)
    ------------------------------------------
    Each test follows the AAA pattern:
    - Arrange: Set up test data and configure mocks
    - Act: Call the method being tested
    - Assert: Verify the expected behavior occurred
    """

    def test_execute_returns_formatted_results_on_success(self, mock_vector_store, sample_search_results):
        """
        Test that execute returns properly formatted results when search succeeds.

        SCENARIO: User asks "What is MCP architecture?" and Claude calls the search tool.
        EXPECTED: The tool returns a formatted string containing the search results
                  that Claude can use to formulate a response.

        WHY THIS MATTERS:
        - This is the "happy path" - the most common successful scenario
        - Verifies the basic contract: query in -> formatted results out
        - Confirms the tool correctly delegates to VectorStore.search()

        COVERAGE:
        - Return type validation (string, not None)
        - Content presence (search results appear in output)
        - VectorStore.search() called with correct parameters
        """
        # Arrange: Configure mock to return sample results
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        tool = CourseSearchTool(mock_vector_store)

        # Act: Execute the search
        result = tool.execute(query="MCP architecture")

        # Assert: Check result and verify mock was called correctly
        assert result is not None
        assert isinstance(result, str)
        assert "MCP architecture" in result or "Build Rich-Context" in result
        mock_vector_store.search.assert_called_once_with(
            query="MCP architecture",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_name_filter(self, mock_vector_store, sample_search_results):
        """
        Test that execute correctly passes course_name filter to VectorStore.

        SCENARIO: Claude determines the user is asking about a specific course
                  and adds a course_name filter to narrow results.
        EXPECTED: The course_name parameter is passed through to VectorStore.search().

        WHY THIS MATTERS:
        - Users often ask about specific courses ("In the MCP course...")
        - Claude uses fuzzy matching to resolve course names
        - We must pass this filter correctly or results will be wrong

        COVERAGE:
        - Parameter passing for course_name filter
        - Verifies filter is not lost or modified in transit
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        # Act: Search with course filter
        result = tool.execute(query="test query", course_name="MCP")

        # Assert: Verify course_name was passed correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, mock_vector_store, sample_search_results):
        """
        Test that execute correctly passes lesson_number filter to VectorStore.

        SCENARIO: User asks about a specific lesson ("In lesson 3...")
                  and Claude extracts and passes the lesson number.
        EXPECTED: The lesson_number parameter is passed through to VectorStore.search().

        WHY THIS MATTERS:
        - Lesson filtering enables precise content retrieval
        - Users often reference specific lessons by number
        - Ensures numerical filters work correctly (not just string filters)

        COVERAGE:
        - Parameter passing for lesson_number filter
        - Integer parameter handling
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        # Act: Search with lesson filter
        result = tool.execute(query="test query", lesson_number=3)

        # Assert: Verify lesson_number was passed correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_all_filters(self, mock_vector_store, sample_search_results):
        """
        Test that execute correctly passes ALL filters when combined.

        SCENARIO: User asks about a specific lesson in a specific course
                  ("In lesson 2 of the MCP course...").
        EXPECTED: Both course_name AND lesson_number are passed to search.

        WHY THIS MATTERS:
        - Combines both filtering mechanisms
        - Verifies no parameter interference when used together
        - Most precise search scenario

        COVERAGE:
        - Combined filter parameter passing
        - Ensures filters don't overwrite each other
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        # Act: Search with both filters
        result = tool.execute(query="test query", course_name="MCP", lesson_number=2)

        # Assert: Verify both filters were passed
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_returns_error_message_on_search_error(self, mock_vector_store, error_search_results):
        """
        Test that execute returns a user-friendly error message when search fails.

        SCENARIO: User asks about a course that doesn't exist. VectorStore returns
                  an error because fuzzy matching couldn't find a match.
        EXPECTED: The error message is included in the tool's response so Claude
                  can communicate the issue to the user.

        WHY THIS MATTERS:
        - Errors must be surfaced, not swallowed silently
        - Claude needs error context to respond appropriately
        - User experience: clear errors are better than confusing responses

        COVERAGE:
        - Error message propagation from VectorStore
        - Graceful error handling (no exceptions thrown)
        """
        # Arrange: Configure mock to return error result
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act: Search for nonexistent course
        result = tool.execute(query="test query", course_name="nonexistent")

        # Assert: Error message appears in output
        assert "No course found matching 'nonexistent'" in result

    def test_execute_returns_no_content_message_on_empty_results(self, mock_vector_store, empty_search_results):
        """
        Test that execute returns appropriate message when no results are found.

        SCENARIO: User asks a valid question but no relevant content exists
                  in the course materials (rare but possible).
        EXPECTED: A clear "no content found" message is returned.

        WHY THIS MATTERS:
        - Empty results != error (search worked, just found nothing)
        - Claude needs to know the difference to respond appropriately
        - Better UX: "I couldn't find info on X" vs generic error

        COVERAGE:
        - Empty result handling (different from error handling)
        - User-friendly messaging for edge case
        """
        # Arrange: Configure mock to return empty results
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act: Search that returns nothing
        result = tool.execute(query="very specific query that returns nothing")

        # Assert: Appropriate empty message
        assert "No relevant content found" in result

    def test_execute_stores_sources_after_successful_search(self, mock_vector_store, sample_search_results):
        """
        Test that execute stores sources for later retrieval (for citations).

        SCENARIO: After a successful search, the UI wants to display source
                  citations showing which lessons the answer came from.
        EXPECTED: The tool stores source information in last_sources property.

        WHY THIS MATTERS:
        - Transparency: Users should know where answers come from
        - Trust: Citations make AI responses more credible
        - Navigation: Users can click through to original content

        COVERAGE:
        - Source storage mechanism
        - Source data structure (must contain 'text' key)
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="MCP")

        # Assert: Sources were stored and have expected structure
        assert len(tool.last_sources) > 0
        assert all("text" in source for source in tool.last_sources)

    def test_execute_includes_lesson_info_in_formatted_output(self, mock_vector_store, sample_search_results):
        """
        Test that formatted output includes lesson information for context.

        SCENARIO: Claude receives search results and needs to understand
                  which lesson each piece of content came from.
        EXPECTED: The formatted output includes "Lesson" references.

        WHY THIS MATTERS:
        - Context helps Claude give better responses ("In lesson 2...")
        - Users can navigate to specific lessons
        - Structured output is more useful than raw text

        COVERAGE:
        - Output formatting includes metadata
        - Lesson information is not lost during formatting
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="test")

        # Assert: Lesson info present in output
        assert "Lesson" in result

    def test_get_tool_definition_returns_valid_schema(self, mock_vector_store):
        """
        Test that get_tool_definition returns a valid Anthropic tool definition.

        SCENARIO: When initializing the RAG system, we need to send tool
                  definitions to Claude API so it knows what tools are available.
        EXPECTED: The definition follows Anthropic's tool schema format.

        WHY THIS MATTERS:
        - Invalid schema = Claude can't use the tool at all
        - Required fields must be correct for tool calls to work
        - This is the contract between our tool and the Anthropic API

        COVERAGE:
        - Tool name is correct ("search_course_content")
        - Required schema fields are present (name, description, input_schema)
        - Required parameters are defined (query is required)

        REFERENCE:
        See Anthropic docs: https://docs.anthropic.com/claude/docs/tool-use
        """
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert: Schema follows Anthropic's expected format
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]


class TestToolManager:
    """
    Test suite for ToolManager - the registry that manages all available tools.

    ToolManager serves as the central hub for tool registration and execution.
    When Claude decides to use a tool, the AIGenerator asks ToolManager to
    execute it by name with the provided arguments.

    RESPONSIBILITIES TESTED:
    1. Tool Registration: Adding tools to the available pool
    2. Tool Execution: Routing execution requests to the right tool
    3. Error Handling: Gracefully handling requests for unknown tools
    4. Definition Export: Providing tool schemas for Claude API

    DESIGN PATTERN:
    ToolManager implements the Registry pattern - it maintains a dictionary
    of tools by name and provides lookup/execution services.
    """

    def test_register_tool_adds_tool_to_manager(self, mock_vector_store):
        """
        Test that register_tool correctly adds a tool to the registry.

        SCENARIO: During system initialization, we register CourseSearchTool
                  so Claude can use it during conversations.
        EXPECTED: The tool is stored in the manager's internal registry.

        WHY THIS MATTERS:
        - If registration fails, Claude can't use the tool
        - Tools must be registered before they can be executed
        - Verifies the basic registration mechanism works

        COVERAGE:
        - Tool registration adds to internal dictionary
        - Tool is accessible by its name
        """
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Act
        manager.register_tool(tool)

        # Assert: Tool is in the registry under its name
        assert "search_course_content" in manager.tools

    def test_execute_tool_calls_correct_tool(self, mock_vector_store, sample_search_results):
        """
        Test that execute_tool routes execution to the correct registered tool.

        SCENARIO: Claude responds with a tool_use block requesting
                  "search_course_content" with query="test".
        EXPECTED: ToolManager finds the right tool and calls its execute method.

        WHY THIS MATTERS:
        - This is the critical path for all tool usage
        - Wrong routing = wrong tool called = broken functionality
        - Verifies the lookup-and-execute mechanism

        COVERAGE:
        - Tool lookup by name works correctly
        - Arguments are passed through to the tool
        - Tool's underlying method (VectorStore.search) is called
        """
        # Arrange: Register tool with working mock
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Act: Execute through manager
        result = manager.execute_tool("search_course_content", query="test")

        # Assert: Tool was called correctly
        assert result is not None
        mock_vector_store.search.assert_called_once()

    def test_execute_tool_returns_error_for_unknown_tool(self):
        """
        Test that execute_tool returns error for unregistered/unknown tools.

        SCENARIO: Due to a bug or malformed request, we try to execute
                  a tool that doesn't exist ("unknown_tool").
        EXPECTED: An error message is returned (not an exception).

        WHY THIS MATTERS:
        - Defensive programming: handle unexpected inputs gracefully
        - Claude might hallucinate tool names (rare but possible)
        - Error message helps debugging

        COVERAGE:
        - Unknown tool handling doesn't crash
        - Error message is informative
        """
        # Arrange: Empty manager with no tools
        manager = ToolManager()

        # Act: Try to execute nonexistent tool
        result = manager.execute_tool("unknown_tool", query="test")

        # Assert: Error message returned, no exception
        assert "not found" in result

    def test_get_tool_definitions_returns_all_registered_tools(self, mock_vector_store):
        """
        Test that get_tool_definitions returns schemas for ALL registered tools.

        SCENARIO: Before calling Claude API, we need to send all tool
                  definitions so Claude knows what tools are available.
        EXPECTED: A list containing the definition for each registered tool.

        WHY THIS MATTERS:
        - Claude can only use tools it knows about
        - Missing definitions = missing functionality
        - This feeds directly into the Claude API call

        COVERAGE:
        - Returns list of all tool definitions
        - Each definition includes required fields
        """
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert: All registered tools included
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
