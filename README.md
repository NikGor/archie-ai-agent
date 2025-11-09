# Archie AI Agent

A sophisticated AI-powered conversational agent built with FastAPI, featuring multiple personas, web search capabilities, and external backend integration.

## Features

- 🤖 **Multiple AI Personas**: Switch between different conversation styles (business, flirty, futurebot)
- 🔍 **Web Search Integration**: Real-time web search capabilities through integrated tools
- 💬 **External Backend Integration**: Works with separate backend service for data persistence
- 🎵 **Voice Support**: Voice assistant capabilities with audio processing
- 🚀 **RESTful API**: Modern FastAPI-based REST API with automatic documentation
- 🐳 **Docker Support**: Containerized deployment with shared network
- 📊 **Rich Metadata**: Enhanced responses with cards, buttons, and interactive elements
- 🔧 **Developer-Friendly**: Poetry dependency management, Makefile automation

## Architecture

This service is part of a microservices architecture:

- **Agent Builder**: Creates AI agents with configurable personas and tools
- **API Controller**: Handles chat requests and integrates with external backend
- **External Backend**: Separate service for data persistence and conversation management
- **Prompt System**: Jinja2-templated prompts for different personas
- **Voice Interface**: Audio input/output capabilities
- **State Management**: Redis-based state storage (planned)

## Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Docker & Docker Compose (optional, for containerized deployment)
- OpenAI API key

## Installation

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd archie-ai-agent
   ```

2. **Install dependencies using Poetry**:
   ```bash
   make install
   # or manually:
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key:
   # OPENAI_API_KEY=your_api_key_here
   ```

4. **Ensure external backend is running**:
   The service requires a separate backend service running at `http://localhost:8002` or configured via `BACKEND_API_URL`.

### Docker Setup

1. **Build and run with Docker Compose**:
   ```bash
   make docker-build
   make docker-run
   ```

The service will be available at `http://localhost:8001` (Docker) or `http://localhost:8000` (local development).

## Usage

### Starting the Server

**Local development**:
```bash
make run          # Production mode
make dev          # Development mode with auto-reload
```

**Docker**:
```bash
make docker-run   # Start with Docker Compose
```

### API Endpoints

#### Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how can you help me today?",
    "role": "user",
    "conversation_id": "test-conversation"
  }'
```

> **Note**: Conversation management endpoints are now handled by the external backend service.

### Interactive Interfaces

**Console Chat**:
```bash
make chat
```

**Voice Assistant**:
```bash
make voice
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_PERSONA=business
DEFAULT_USER_NAME=User
BACKEND_API_URL=http://localhost:8002
REDIS_HOST=redis
REDIS_PORT=6379
```

### Available Personas

The system supports multiple conversation personas:

- **business**: Professional and formal communication style
- **flirty**: Playful and engaging conversation style  
- **futurebot**: Forward-thinking, tech-focused personality

Personas are defined in `app/prompts/persona_*.jinja2` files.

## Development

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make run           # Start production server
make dev           # Start development server with auto-reload
make chat          # Start console chat interface
make voice         # Start voice assistant
make test          # Run tests (placeholder)
make clean         # Clean cache and temporary files
make docker-build  # Build Docker image
make docker-run    # Run with Docker Compose
make docker-stop   # Stop Docker containers
make setup         # Complete development environment setup
make status        # Check if server is running
make api-test      # Test API with sample request
```

### Adding New Personas

1. Create a new prompt template in `app/prompts/persona_yourname.jinja2`
2. Update the `DEFAULT_PERSONA` in your `.env` file if needed
3. The agent builder will automatically load the new persona

### Extending Functionality

- **Add new tools**: Extend the tools in `app/tools.py`
- **Modify responses**: Update models in `app/models.py`
- **Add endpoints**: Extend `app/endpoints.py`
- **Custom prompts**: Modify templates in `app/prompts/`

## API Response Format

The API returns rich responses with metadata:

```json
{
  "message_id": "uuid",
  "role": "assistant",
  "text": "Response text",
  "text_format": "markdown",
  "metadata": {
    "cards": [
      {
        "title": "Information Card",
        "text": "Card content",
        "options": {
          "buttons": [
            {
              "text": "Action Button",
              "command": "action_command"
            }
          ]
        }
      }
    ]
  },
  "created_at": "2025-09-28T10:00:00Z",
  "conversation_id": "conversation_uuid"
}
```

## Dependencies

Main dependencies:
- **FastAPI**: Modern web framework for building APIs
- **openai-agents**: AI agent framework with OpenAI integration
- **Jinja2**: Template engine for prompts
- **Pydantic**: Data validation and settings management
- **httpx**: HTTP client for external API integration
- **Poetry**: Dependency management and packaging

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Check the API documentation at `/docs` when running the server
- Review the Makefile for available development commands

## Roadmap

- [ ] Unit test coverage
- [ ] Authentication and authorization
- [ ] WebSocket support for real-time chat
- [ ] Multi-language support
- [ ] Enhanced voice capabilities
- [ ] Plugin system for custom tools
- [ ] Conversation export/import
- [ ] Performance monitoring and analytics

---

Built with ❤️ using FastAPI, OpenAI, and modern Python tools.
