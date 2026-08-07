# Contributing to Memex

Thank you for your interest in contributing to Memex! This document provides guidelines and information about contributing to this project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/memex.git
   cd memex
   ```
3. **Install dependencies**:
   ```bash
   uv sync --extra dev --extra test
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation if needed

### 3. Run Quality Checks
```bash
# Run linter
make lint

# Auto-format code
make fmt

# Run tests
make test

# Run all checks
make lint && make test
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
# Or for bug fixes:
git commit -m "fix: describe the bug fix"
```

Use [Conventional Commits](https://www.conventionalcommits.org/) format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

### 5. Push and Create a PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference any related issues
- Include screenshots if applicable

## Code Style

### Python
- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Line length**: 120 characters
- **Import sorting**: `ruff check --select I`
- **Type hints**: Required for all functions
- **Docstrings**: Google style for public functions

### Example
```python
def process_document(
    file_path: str,
    chunk_size: int = 1024,
    overlap: int = 128,
) -> list[dict[str, Any]]:
    """Process a document into chunks.
    
    Args:
        file_path: Path to the document file.
        chunk_size: Target size for each chunk in tokens.
        overlap: Number of tokens to overlap between chunks.
        
    Returns:
        List of chunk dictionaries with content and metadata.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the document is empty.
    """
    pass
```

## Testing

### Running Tests
```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_config.py -v

# Run with coverage
pytest tests/ --cov=memex --cov-report=html

# Run only failed tests
pytest --lf
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use fixtures in `tests/fixtures/` for test data
- Follow the naming convention: `test_<function_name>.py`
- Use descriptive test names: `test_process_document_handles_empty_file`

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Update CHANGELOG.md with your changes
- Create design specs in `docs/superpowers/specs/` for major features

## Reporting Issues

### Bug Reports
Use GitHub Issues with the "bug" label. Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### Feature Requests
Use GitHub Issues with the "enhancement" label. Include:
- Clear description of the feature
- Use case and motivation
- Possible implementation approach

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Celebrate contributions of all sizes

## Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search GitHub Issues
3. Open a new issue with the "question" label

Thank you for contributing to Memex! 🚀
