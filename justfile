# Justfile for CNNParted
# Run 'just' to see available commands

# Default recipe to display help
default:
    @just --list

# Install CNNParted using uv (full installation)
install:
    @echo "🚀 Installing CNNParted..."
    ./install.sh

# Quick development setup
dev-setup:
    @echo "🔧 Setting up development environment..."
    ./dev_setup.sh

# Activate development environment
dev:
    @echo "Activating development environment..."
    @echo "Run: source dev_activate.sh"

# Run tests
test:
    @echo "Running tests..."
    uv run pytest

# Format code
format:
    @echo "Formatting code..."
    uv run black .

# Lint code
lint:
    @echo "Linting code..."
    uv run flake8 .

# Type check
typecheck:
    @echo "Type checking..."
    uv run mypy .

# Run a quick example
example:
    @echo "Running example..."
    uv run cnnparted examples/squeezenet1_1.yaml test_run

# Clean build artifacts
clean:
    @echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Build package
build:
    @echo "Building package..."
    uv build

# Install pre-commit hooks
setup-hooks:
    @echo "Setting up pre-commit hooks..."
    uv run pre-commit install

# Update dependencies
update:
    @echo "Updating dependencies..."
    uv pip compile pyproject.toml -o requirements.lock
    uv pip sync requirements.lock
