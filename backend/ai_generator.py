import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
   - Use for questions about specific topics, concepts, or detailed educational content
   - Returns relevant text excerpts from course lessons

2. **get_course_outline**: Get complete course structure and lesson list
   - Use for questions about course structure, syllabus, available lessons, or course overview
   - Returns: course title, course link, and all lessons (number and title for each)
   - Use this when users ask "what lessons are in...", "show me the outline of...", "what does [course] cover?"

Tool Usage Guidelines:
- **Up to two sequential tool calls per query** when needed
- Use a second tool call only when the first result is insufficient
- You may combine tools: e.g., get outline first, then search specific content
- For outline/structure questions: use get_course_outline
- For content/topic questions: use search_course_content
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results" or "based on the outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls. Each round:
        1. Send messages to Claude (with tools if rounds remain)
        2. If Claude uses a tool, execute it and continue
        3. If Claude responds with text or max rounds reached, return response

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history with user query
        messages = [{"role": "user", "content": query}]
        round_count = 0

        # Tool calling loop - up to MAX_TOOL_ROUNDS
        while round_count < self.MAX_TOOL_ROUNDS:
            # Build API parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Include tools if available and we have rounds remaining
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # If no tool use, return the text response
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            # Execute tools and collect results
            tool_results, _ = self._execute_tools(response, tool_manager)

            # Add assistant's tool use response to message history
            messages.append({"role": "assistant", "content": response.content})

            # Add tool results to message history
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            round_count += 1

        # Max rounds reached - get final response without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools(self, response, tool_manager) -> tuple:
        """
        Execute tool calls from a response.

        Args:
            response: The API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, success boolean).
            success is False if any tool raised an exception.
        """
        tool_results = []
        success = True

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    result = f"Error executing {content_block.name}: {str(e)}"
                    success = False

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": result
                })

        return tool_results, success
