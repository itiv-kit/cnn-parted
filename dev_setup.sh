#!/bin/bash

# CNNParted Quick Development Setup
# Minimal setup for development without heavy external tools

set -e

echo "🚀 Quick development setup for CNNParted..."

# Check basic requirements
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Run from CNNParted root directory"
    exit 1
fi

# Install uv if needed
if ! command -v uv >/dev/null 2>&1; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Quick setup
echo "📦 Setting up development environment..."
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Only get essential submodules
echo "📥 Getting essential tools..."
git submodule update --init tools/accelergy
git submodule update --init tools/accelergy-aladdin-plug-in
git submodule update --init tools/dnn-model-exploration

# Install just the Python tools (no building)
echo "⚙️  Installing Python tools..."
uv pip install pyyaml setuptools
uv pip install -e tools/accelergy/
uv pip install tools/accelergy-aladdin-plug-in/
uv pip install -e tools/dnn-model-exploration/

# Simple activation script
cat > activate_dev.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "✅ CNNParted dev environment activated!"
EOF

chmod +x activate_dev.sh

echo ""
echo "🎉 Development setup complete!"
echo "   Use: source activate_dev.sh"
echo "   Note: External tools (CACTI, Timeloop) not built in dev mode"
echo "   Run ./install.sh for full installation"
