"""
RAG Chatbot Test Suite
=======================

This package contains unit and integration tests for the RAG chatbot backend.

TEST MODULES:
-------------
- conftest.py: Shared pytest fixtures (MockConfig, mock_vector_store, sample results)
- test_course_search_tool.py: Tests for CourseSearchTool and ToolManager
- test_ai_generator.py: Tests for AIGenerator's tool calling behavior
- test_rag_system.py: Integration tests for the complete RAG pipeline

RUNNING TESTS:
--------------
From the backend directory:

    # Run all tests
    uv run pytest tests/ -v

    # Run specific test file
    uv run pytest tests/test_course_search_tool.py -v

    # Run with coverage
    uv run pytest tests/ --cov=. --cov-report=html

TEST PHILOSOPHY:
----------------
1. Unit tests mock external dependencies (APIs, databases)
2. Tests follow Arrange-Act-Assert (AAA) pattern
3. Each test has a clear SCENARIO, EXPECTED, and WHY THIS MATTERS documentation
4. Fixtures in conftest.py are shared across all test modules

ADDING NEW TESTS:
-----------------
1. Create a new test_*.py file in this directory
2. Import shared fixtures from conftest.py (they're auto-discovered)
3. Group related tests in classes (e.g., TestMyFeature)
4. Document each test with SCENARIO/EXPECTED/WHY THIS MATTERS pattern

For more details, see the docstrings in each test module.
"""
