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
```

The web interface is available at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for course materials. The system uses ChromaDB for vector storage and Anthropic Claude for AI responses.

### Data Flow

1. **Document Ingestion**: Course documents (PDF, DOCX, TXT) from `../docs` are processed on startup
2. **Text Chunking**: Documents are split into sentence-based chunks (800 chars, 100 overlap) via `DocumentProcessor`
3. **Vector Storage**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB (two collections: `course_catalog` for metadata, `course_content` for searchable chunks)
4. **Query Processing**: User queries trigger AI-driven tool calls to search the vector store
5. **Response Generation**: Claude synthesizes search results into responses

### Key Components

- **`RAGSystem`** (`rag_system.py`): Main orchestrator that wires together all components
- **`VectorStore`** (`vector_store.py`): ChromaDB wrapper with semantic search and course name resolution
- **`AIGenerator`** (`ai_generator.py`): Handles Claude API calls with tool execution loop
- **`CourseSearchTool`** (`search_tools.py`): Anthropic tool definition for searching course content
- **`SessionManager`** (`session_manager.py`): Manages conversation history per session

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
