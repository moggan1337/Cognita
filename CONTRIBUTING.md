# Contributing to Cognita

Thank you for your interest in contributing to Cognita!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cognita.git
   cd Cognita
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

```bash
pytest tests/
```

## Code Style

We use:
- Black for formatting
- isort for imports
- mypy for type checking

```bash
black src/
isort src/
mypy src/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Questions?

Open an issue on GitHub for any questions.
