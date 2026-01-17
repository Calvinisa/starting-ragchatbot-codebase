# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Alternative: use the shell script (requires Git Bash on Windows)
./run.sh

# Run tests (from project root)
cd backend && uv run pytest tests/ -v

# Run tests with coverage
cd backend && uv run pytest tests/ --cov=. --cov-report=html

# Format code with black
uv run black backend/ main.py

# Check formatting without changes
uv run black --check backend/ main.py

# Run quality checks script (requires Git Bash on Windows)
./quality.sh          # Format code
./quality.sh --check  # Check only
```

The web interface is available at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for course materials. The system uses ChromaDB for vector storage and Anthropic Claude for AI responses.

### Data Flow

1. **Document Ingestion**: Course documents (`.txt` files) from `../docs` are processed on startup
2. **Text Chunking**: Documents are split into sentence-based chunks (800 chars, 100 overlap) via `DocumentProcessor`
3. **Vector Storage**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB
4. **Query Processing**: User queries trigger AI-driven tool calls to search the vector store
5. **Response Generation**: Claude synthesizes search results into responses

### ChromaDB Collections

The system uses two separate ChromaDB collections:
- **`course_catalog`**: Stores course metadata (title, instructor, lessons as JSON) for semantic course name resolution
- **`course_content`**: Stores searchable text chunks with metadata (course_title, lesson_number, chunk_index)

### Tool Execution Loop

The `AIGenerator` implements an agentic loop for tool calling:
1. Send user query to Claude with `CourseSearchTool` definition
2. If Claude returns `stop_reason == "tool_use"`, execute the requested search
3. Send tool results back to Claude for final response synthesis
4. Return the text response to the user

### Key Components

- **`RAGSystem`** (`rag_system.py`): Main orchestrator that wires together all components
- **`VectorStore`** (`vector_store.py`): ChromaDB wrapper with semantic search and fuzzy course name resolution
- **`AIGenerator`** (`ai_generator.py`): Handles Claude API calls with tool execution loop
- **`CourseSearchTool`** (`search_tools.py`): Anthropic tool definition for searching course content
- **`SessionManager`** (`session_manager.py`): In-memory conversation history per session

### Document Format

Course documents expect this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [url]
[content]

Lesson 1: [lesson title]
...
```

### API Endpoints

- `POST /api/query` - Process a question with RAG (accepts `query` and optional `session_id`)
- `GET /api/courses` - Get course catalog statistics

## Configuration

Settings are in `backend/config.py`. Key parameters:
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation turns retained
- `CHROMA_PATH`: `./chroma_db`

Requires `ANTHROPIC_API_KEY` in `.env` file at project root.

## UI Development Workflow

When making frontend changes, use the Playwright MCP server to automatically test and verify changes:

1. **Navigate to the app**: Use `browser_navigate` to open `http://127.0.0.1:8000`
2. **Inspect current state**: Use `browser_snapshot` to get the accessibility tree of the page
3. **Make code changes**: Edit CSS/HTML/JS files as needed
4. **Verify visually**: Use `browser_take_screenshot` to capture the result
5. **Test interactions**: Use `browser_click`, `browser_type`, etc. to test functionality

### Useful Playwright MCP Commands

- `browser_navigate` - Load a URL
- `browser_snapshot` - Get page structure (preferred over screenshots for understanding layout)
- `browser_take_screenshot` - Capture visual state
- `browser_click` - Click elements by ref from snapshot
- `browser_type` - Enter text in input fields
- `browser_evaluate` - Run JavaScript (useful for injecting test CSS)
- `browser_console_messages` - Check for JS errors

### Quick CSS Testing

To preview CSS changes before committing, inject styles via `browser_evaluate`:
```javascript
() => {
  const style = document.createElement('style');
  style.textContent = `.my-class { color: red !important; }`;
  document.head.appendChild(style);
}
```
