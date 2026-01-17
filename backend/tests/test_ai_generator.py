"""
Tests for AIGenerator Tool Calling Behavior
=============================================

This module tests the AIGenerator class, which is responsible for communicating
with the Claude API and managing the tool execution loop. This is the core of
how the RAG system generates AI-powered responses.

WHAT IS AIGenerator?
--------------------
AIGenerator wraps the Anthropic API client and implements an "agentic loop" for
tool calling. When Claude decides to use a tool (like searching course content),
AIGenerator:
1. Receives Claude's tool_use response
2. Executes the requested tool via ToolManager
3. Sends the tool results back to Claude
4. Returns Claude's final response to the user

THE AGENTIC LOOP:
-----------------
    User Query
         |
         v
    Claude API (with tools) -----> Response includes tool_use?
         ^                              |
         |                         YES  |  NO
         |                              |   |
    Tool Result                         v   v
         ^                     Execute Tool  Return Text
         |                              |
         +---------- Send Result -------+

TEST COVERAGE OVERVIEW:
-----------------------
TestAIGeneratorToolCalling (6 tests):
    - API call includes tool definitions
    - Tool use response triggers execution
    - Correct tool is executed with correct arguments
    - Tool results are properly formatted and sent back
    - Works without tools (direct response)
    - Conversation history is included in context

TestAIGeneratorToolSelection (2 tests):
    - System prompt includes search tool instructions
    - System prompt includes outline tool instructions

MOCKING STRATEGY:
-----------------
These tests mock the Anthropic API client to avoid:
1. Real API calls (cost and rate limits)
2. Network dependencies (tests should be fast and reliable)
3. API key requirements (tests should run anywhere)

The mock responses simulate Claude's actual response format including
stop_reason, content blocks, and tool_use blocks.

FIXTURES USED:
--------------
    - mock_vector_store (from conftest.py): Mock VectorStore
    - sample_search_results (from conftest.py): Test search results

LOCAL FIXTURES:
---------------
    - mock_anthropic_client: Mock Anthropic API client
    - ai_generator: AIGenerator instance with mocked client
    - mock_tool_manager: ToolManager with registered CourseSearchTool

RUNNING THESE TESTS:
--------------------
    cd backend
    uv run pytest tests/test_ai_generator.py -v
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


@dataclass
class MockToolUseBlock:
    """
    Mock for Anthropic tool_use content block.

    When Claude decides to use a tool, it returns a content block with type="tool_use".
    This dataclass simulates that response for testing without calling the real API.

    Real Anthropic Response Example:
    {
        "type": "tool_use",
        "id": "toolu_01abc123",
        "name": "search_course_content",
        "input": {"query": "MCP architecture"}
    }

    Attributes:
        type: Always "tool_use" for tool calls
        id: Unique identifier for this tool use (used to match results)
        name: The tool name Claude wants to call
        input: Dictionary of arguments for the tool

    Usage:
        tool_block = MockToolUseBlock(input={"query": "custom query"})
        mock_response.content = [tool_block]
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

    When Claude returns a text response (not a tool call), it uses a text block.
    This is the final response that gets shown to the user.

    Real Anthropic Response Example:
    {
        "type": "text",
        "text": "Based on the course materials, MCP architecture is..."
    }

    Attributes:
        type: Always "text" for text responses
        text: The actual response text from Claude

    Usage:
        text_block = MockTextBlock(text="Custom response")
        mock_response.content = [text_block]
    """
    type: str = "text"
    text: str = "Here is the answer based on the search."


class TestAIGeneratorToolCalling:
    """
    Test suite for AIGenerator tool calling functionality.

    This class tests the agentic loop - the process by which Claude can request
    tool execution and receive results. This is the core mechanism that enables
    RAG: Claude searches for information, gets results, and synthesizes a response.

    KEY CONCEPTS TESTED:
    1. Tool definitions are sent to Claude API
    2. stop_reason="tool_use" triggers tool execution
    3. Tool results are formatted correctly for Claude
    4. The loop continues until stop_reason="end_turn"
    """

    @pytest.fixture
    def mock_anthropic_client(self):
        """
        Create a mock Anthropic client.

        This replaces the real anthropic.Anthropic client so we can:
        - Control what responses Claude "returns"
        - Verify what parameters were sent to the API
        - Test without API keys or network access

        Returns:
            Mock: A mock object that will be configured per-test
        """
        return Mock()

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """
        Create AIGenerator with mocked client.

        This fixture patches the anthropic.Anthropic constructor so when
        AIGenerator creates a client, it gets our mock instead.

        Returns:
            AIGenerator: Instance ready for testing with mocked API

        Note:
            We also directly assign generator.client = mock_anthropic_client
            to ensure the mock is properly connected.
        """
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def mock_tool_manager(self, mock_vector_store, sample_search_results):
        """
        Create a mock tool manager with CourseSearchTool registered.

        This creates a real ToolManager with a real CourseSearchTool,
        but the CourseSearchTool uses a mock VectorStore. This tests
        the integration between AIGenerator, ToolManager, and CourseSearchTool.

        Returns:
            ToolManager: Manager with search tool ready to execute

        Note:
            The mock_vector_store is configured to return sample_search_results,
            so executing the search tool will return predictable test data.
        """
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        return manager

    def test_generate_response_calls_api_with_tools(self, ai_generator, mock_tool_manager):
        """
        Test that generate_response includes tools in API call when provided.

        SCENARIO: RAGSystem calls generate_response with tool definitions.
        EXPECTED: The tools parameter is passed to the Claude API.

        WHY THIS MATTERS:
        - If tools aren't sent, Claude can't use them
        - This is the connection between our tool definitions and Claude
        - Verifies the API call is constructed correctly

        COVERAGE:
        - Tools parameter is present in API call
        - Correct tool definitions are passed (not modified)
        """
        # Arrange: Set up mock to return simple text response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MockTextBlock()]
        ai_generator.client.messages.create.return_value = mock_response

        tools = mock_tool_manager.get_tool_definitions()

        # Act: Generate response with tools
        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert: API was called with tools
        call_args = ai_generator.client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools

    def test_generate_response_handles_tool_use_response(self, ai_generator, mock_tool_manager):
        """
        Test that generate_response correctly handles tool_use stop_reason.

        SCENARIO: Claude decides to search for information before answering.
                  First API call returns tool_use, second returns final answer.
        EXPECTED: Two API calls are made, and the final text response is returned.

        WHY THIS MATTERS:
        - This is the core agentic loop behavior
        - stop_reason="tool_use" must trigger tool execution
        - The loop must continue until stop_reason="end_turn"

        COVERAGE:
        - Multiple API calls (loop behavior)
        - Correct handling of tool_use stop_reason
        - Final text response is extracted and returned

        FLOW TESTED:
        1. First call: Claude returns tool_use
        2. Tool is executed (mocked)
        3. Second call: Claude returns text with tool results
        4. Text is returned to caller
        """
        # Arrange: Set up two responses - tool_use then text
        tool_use_block = MockToolUseBlock()

        # First response: Claude wants to use a tool
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        # Second response: Claude gives final answer
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="MCP is a protocol for...")]

        # Configure mock to return responses in sequence
        ai_generator.client.messages.create.side_effect = [first_response, second_response]

        tools = mock_tool_manager.get_tool_definitions()

        # Act
        result = ai_generator.generate_response(
            query="What is MCP architecture?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert: Loop made 2 calls, returned final text
        assert ai_generator.client.messages.create.call_count == 2
        assert result == "MCP is a protocol for..."

    def test_generate_response_executes_correct_tool(self, ai_generator, mock_vector_store, sample_search_results):
        """
        Test that the correct tool is executed based on Claude's response.

        SCENARIO: Claude requests search_course_content with specific arguments.
        EXPECTED: VectorStore.search is called with those exact arguments.

        WHY THIS MATTERS:
        - Claude's tool choice must be respected
        - Arguments must be passed correctly
        - Wrong tool or wrong arguments = wrong results

        COVERAGE:
        - Tool execution based on name from response
        - Argument extraction from tool_use block
        - Arguments passed to underlying search method

        FLOW TESTED:
        1. Claude returns tool_use for search_course_content
        2. AIGenerator extracts name and input
        3. ToolManager.execute_tool is called
        4. CourseSearchTool.execute runs
        5. VectorStore.search is called with extracted args
        """
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Claude's tool request includes specific arguments
        tool_use_block = MockToolUseBlock(
            input={"query": "MCP architecture", "course_name": "MCP"}
        )

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock()]

        ai_generator.client.messages.create.side_effect = [first_response, second_response]

        # Act
        result = ai_generator.generate_response(
            query="Tell me about MCP architecture",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Assert: VectorStore.search was called with Claude's arguments
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["query"] == "MCP architecture"

    def test_generate_response_passes_tool_results_back_to_claude(self, ai_generator, mock_tool_manager):
        """
        Test that tool results are correctly passed back to Claude.

        SCENARIO: After executing a tool, the results must be sent to Claude
                  in the correct format so it can use them in its response.
        EXPECTED: Second API call includes properly formatted tool_result message.

        WHY THIS MATTERS:
        - Claude needs tool results to generate informed responses
        - Wrong format = Claude can't understand the results
        - The tool_use_id must match for Claude to correlate request/response

        COVERAGE:
        - Message sequence: user -> assistant (tool_use) -> user (tool_result)
        - tool_result content block format
        - tool_use_id correlation

        MESSAGE FORMAT TESTED:
        messages = [
            {"role": "user", "content": "What is MCP?"},
            {"role": "assistant", "content": [tool_use_block]},
            {"role": "user", "content": [{"type": "tool_result", ...}]}
        ]
        """
        # Arrange
        tool_use_block = MockToolUseBlock()

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock()]

        ai_generator.client.messages.create.side_effect = [first_response, second_response]

        # Act
        result = ai_generator.generate_response(
            query="What is MCP?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Assert: Check second API call's messages parameter
        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Message sequence: user query, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"  # tool_result comes from "user"

        # Verify tool result format
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"  # Matches MockToolUseBlock.id

    def test_generate_response_without_tools_returns_direct_response(self, ai_generator):
        """
        Test that generate_response works without tools (no RAG).

        SCENARIO: Called without tools parameter - just a simple Q&A.
        EXPECTED: Claude responds directly without any tool execution.

        WHY THIS MATTERS:
        - System should work even when tools aren't needed
        - Backwards compatibility with simple chat
        - Some queries don't need RAG

        COVERAGE:
        - No tools parameter handling
        - Direct text response extraction
        - No agentic loop triggered
        """
        # Arrange: Simple response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MockTextBlock(text="Direct answer")]
        ai_generator.client.messages.create.return_value = mock_response

        # Act: Call without tools
        result = ai_generator.generate_response(query="What is Python?")

        # Assert: Direct response returned
        assert result == "Direct answer"

    def test_generate_response_includes_conversation_history(self, ai_generator):
        """
        Test that conversation history is included in system prompt.

        SCENARIO: User is having a multi-turn conversation. Previous messages
                  should be included for context.
        EXPECTED: System prompt includes "Previous conversation" section.

        WHY THIS MATTERS:
        - Multi-turn conversations need context
        - Without history, Claude can't understand "it" or "that"
        - Better responses come from understanding conversation flow

        COVERAGE:
        - conversation_history parameter is processed
        - History appears in system prompt
        - History format is preserved

        EXAMPLE HISTORY:
            "User: Hello
             Assistant: Hi there!"
        """
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MockTextBlock()]
        ai_generator.client.messages.create.return_value = mock_response

        history = "User: Hello\nAssistant: Hi there!"

        # Act: Call with conversation history
        result = ai_generator.generate_response(
            query="Follow up question",
            conversation_history=history
        )

        # Assert: History included in system prompt
        call_args = ai_generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation" in system_content
        assert history in system_content


class TestAIGeneratorToolSelection:
    """
    Test that AIGenerator's system prompt correctly guides Claude to use tools.

    The system prompt is crucial for proper tool usage. It tells Claude:
    - What tools are available
    - When to use each tool
    - How to interpret tool results

    These tests verify the system prompt contains proper instructions
    without testing actual Claude behavior (that requires integration tests).
    """

    @pytest.fixture
    def ai_generator(self):
        """
        Create AIGenerator with mocked client for system prompt inspection.

        Returns:
            AIGenerator: Instance with accessible SYSTEM_PROMPT attribute
        """
        mock_client = Mock()
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            generator.client = mock_client
            return generator

    def test_system_prompt_contains_search_tool_instructions(self, ai_generator):
        """
        Test that system prompt includes instructions for search_course_content tool.

        SCENARIO: We need Claude to know when and how to use the search tool.
        EXPECTED: System prompt mentions the tool by name and describes content searching.

        WHY THIS MATTERS:
        - Without instructions, Claude might not use the tool appropriately
        - Tool name must be exact for Claude to call it correctly
        - "content" keyword helps Claude understand the tool's purpose

        COVERAGE:
        - Tool name present in prompt
        - Content-related keywords present
        """
        assert "search_course_content" in ai_generator.SYSTEM_PROMPT
        assert "content" in ai_generator.SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_outline_tool_instructions(self, ai_generator):
        """
        Test that system prompt includes instructions for get_course_outline tool.

        SCENARIO: We need Claude to know when to use the outline tool (for course structure queries).
        EXPECTED: System prompt mentions the outline tool and describes its purpose.

        WHY THIS MATTERS:
        - Outline queries are different from content queries
        - Claude needs to know both tools exist and their purposes
        - "outline" keyword helps Claude match user intent

        COVERAGE:
        - Outline tool name present in prompt
        - Outline-related keywords present
        """
        assert "get_course_outline" in ai_generator.SYSTEM_PROMPT
        assert "outline" in ai_generator.SYSTEM_PROMPT.lower()


class TestAIGeneratorMultiRoundToolCalling:
    """
    Test suite for sequential tool calling (up to 2 rounds).

    This class tests the multi-round agentic loop where Claude can make
    up to MAX_TOOL_ROUNDS (2) sequential tool calls before being forced
    to provide a final response.

    KEY CONCEPTS TESTED:
    1. Tools remain available across rounds (until max)
    2. Message history accumulates correctly
    3. Termination conditions work as expected
    4. Error handling is graceful
    """

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        return Mock()

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator with mocked client."""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def mock_tool_manager(self, mock_vector_store, sample_search_results):
        """Create a mock tool manager with CourseSearchTool registered."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        return manager

    def test_two_sequential_tool_calls_succeed(self, ai_generator, mock_tool_manager):
        """
        Test that two sequential tool calls work correctly.

        SCENARIO: Claude calls search_course_content in round 1, then again in round 2.
        EXPECTED: 3 API calls made (initial + 2 rounds), both tools executed.

        FLOW:
        1. First call: Claude returns tool_use
        2. Tool executed, results sent back
        3. Second call: Claude returns tool_use again
        4. Tool executed, results sent back
        5. Third call: Claude returns final text
        """
        # Round 1: First tool call
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [MockToolUseBlock(id="tool_1")]

        # Round 2: Second tool call
        second_response = Mock()
        second_response.stop_reason = "tool_use"
        second_response.content = [MockToolUseBlock(id="tool_2", input={"query": "follow-up search"})]

        # Final: Text response
        third_response = Mock()
        third_response.stop_reason = "end_turn"
        third_response.content = [MockTextBlock(text="Based on both searches, here is the answer.")]

        ai_generator.client.messages.create.side_effect = [first_response, second_response, third_response]

        tools = mock_tool_manager.get_tool_definitions()
        result = ai_generator.generate_response(
            query="Compare topics across courses",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert: 3 API calls made
        assert ai_generator.client.messages.create.call_count == 3
        assert result == "Based on both searches, here is the answer."

    def test_second_round_no_tool_call_terminates_early(self, ai_generator, mock_tool_manager):
        """
        Test that loop terminates when Claude responds without tool use in round 2.

        SCENARIO: Claude uses a tool in round 1, then provides direct text in round 2.
        EXPECTED: 2 API calls made, response returned from second call.
        """
        # Round 1: Tool call
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [MockToolUseBlock()]

        # Round 2: Direct text response (no tool use)
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="Answer after one search.")]

        ai_generator.client.messages.create.side_effect = [first_response, second_response]

        tools = mock_tool_manager.get_tool_definitions()
        result = ai_generator.generate_response(
            query="Simple question",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert: Only 2 API calls (terminated early)
        assert ai_generator.client.messages.create.call_count == 2
        assert result == "Answer after one search."

    def test_max_rounds_enforced(self, ai_generator, mock_tool_manager):
        """
        Test that tool calling stops after MAX_TOOL_ROUNDS even if Claude wants more.

        SCENARIO: Claude would keep calling tools, but we cap at 2 rounds.
        EXPECTED: After 2 tool rounds, final call is made without tools.
        """
        # Both rounds: Claude wants to use tools
        tool_response_1 = Mock()
        tool_response_1.stop_reason = "tool_use"
        tool_response_1.content = [MockToolUseBlock(id="tool_1")]

        tool_response_2 = Mock()
        tool_response_2.stop_reason = "tool_use"
        tool_response_2.content = [MockToolUseBlock(id="tool_2")]

        # Final forced response (after max rounds)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MockTextBlock(text="Forced final answer.")]

        ai_generator.client.messages.create.side_effect = [tool_response_1, tool_response_2, final_response]

        tools = mock_tool_manager.get_tool_definitions()
        result = ai_generator.generate_response(
            query="Complex multi-step query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert: 3 API calls total
        assert ai_generator.client.messages.create.call_count == 3

        # Assert: Final call has no tools (forced completion)
        final_call_args = ai_generator.client.messages.create.call_args_list[2]
        assert "tools" not in final_call_args.kwargs

        assert result == "Forced final answer."

    def test_tool_failure_graceful_response(self, ai_generator):
        """
        Test that tool execution failure is handled gracefully.

        SCENARIO: Tool execution raises an exception.
        EXPECTED: Error message is sent to Claude, Claude responds gracefully.
        """
        # Create a tool manager that will fail
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        # Round 1: Tool call
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [MockToolUseBlock(name="test_tool")]

        # Round 2: Claude responds after seeing error
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [MockTextBlock(text="I encountered an error but here is what I know.")]

        ai_generator.client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Assert: Error was handled gracefully
        assert "I encountered an error" in result

        # Assert: Error message was passed to Claude
        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]
        tool_result_message = messages[2]["content"][0]
        assert "Error executing test_tool" in tool_result_message["content"]

    def test_message_history_accumulates_correctly(self, ai_generator, mock_tool_manager):
        """
        Test that message history is correctly built across rounds.

        SCENARIO: Two tool calls are made sequentially.
        EXPECTED: Final API call contains full message history with all exchanges.

        MESSAGE STRUCTURE:
        [user query] -> [assistant tool_use_1] -> [user tool_result_1]
                     -> [assistant tool_use_2] -> [user tool_result_2]
        """
        # Round 1
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [MockToolUseBlock(id="tool_1")]

        # Round 2
        second_response = Mock()
        second_response.stop_reason = "tool_use"
        second_response.content = [MockToolUseBlock(id="tool_2")]

        # Final
        third_response = Mock()
        third_response.stop_reason = "end_turn"
        third_response.content = [MockTextBlock()]

        ai_generator.client.messages.create.side_effect = [first_response, second_response, third_response]

        tools = mock_tool_manager.get_tool_definitions()
        ai_generator.generate_response(
            query="Multi-step question",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Check final API call's message structure
        final_call_args = ai_generator.client.messages.create.call_args_list[2]
        messages = final_call_args.kwargs["messages"]

        # Should have 5 messages:
        # 1. user query
        # 2. assistant tool_use_1
        # 3. user tool_result_1
        # 4. assistant tool_use_2
        # 5. user tool_result_2
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

        # Verify tool result IDs match
        assert messages[2]["content"][0]["tool_use_id"] == "tool_1"
        assert messages[4]["content"][0]["tool_use_id"] == "tool_2"
