# CNNParted

## About
CNNParted is a framework for hardware-aware design space exploration of CNN inference partitioning in embedded AI applications. Thereby, not only the hardware deployed in the nodes is considered but also the impact of the link on system energy consumption and inference latency.

The framework currently includes a custom Ethernet model as well as hardware accelerators models taken from Timeloop example repository (Simba, Eyeriss and Simple OS Array).

## 🚀 Quick Installation with uv

We've modernized the installation process using [uv](https://github.com/astral-sh/uv), Python's fastest package installer and resolver.

### Prerequisites
- Python 3.11 or higher
- Git
- Build tools (gcc, make, cmake)

### Option 1: Full Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/itiv-kit/cnn-parted.git
cd cnn-parted

# Run the automated installation script
./install.sh
```

This script will:
- Install `uv` if not present
- Create a virtual environment
- Install all dependencies
- Build external tools (CACTI, Ramulator, VAMPIRE, Timeloop)
- Set up git submodules
- Create an activation script

### Option 2: Development Setup (Quick)
For developers who want to get started quickly without building all external tools:

```bash
# Quick development setup
./dev_setup.sh

# Activate development environment
source dev_activate.sh
```

### Option 3: Manual Installation with uv
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install CNNParted (core dependencies only)
uv pip install -e ".[dev]"

# Optionally install NVIDIA dependencies (may fail without CUDA)
# uv pip install nvidia-pyindex pytorch-quantization

# Note: Research tools (accelergy, timeloop, etc.) are not in PyPI
# Use ./install.sh for complete installation including research tools
```

## 🏃 Running CNNParted

After installation, activate the environment and run:

```bash
# Activate environment
source activate_env.sh

# Run an example (note: may encounter simulator limitations with some models)
cnnparted examples/squeezenet1_1.yaml test_run

# Test installation
cnnparted --help
```

**Note:** Some models may encounter simulator limitations (e.g., unsupported layer types like "concat" in MNSIM). This is expected behavior - the external simulators have specific supported layer types. You can try different models or modify the configuration to work around these limitations.

## 🛠️ Development

### Using just (Recommended)
We provide a `justfile` for common development tasks:

```bash
# Install just if not already installed
cargo install just

# See available commands
just

# Quick development setup
just dev-setup

# Run tests
just test

# Format code
just format

# Run example
just example
```

### Manual Development Commands
```bash
# Run tests
uv run pytest

# Format code
uv run black .

# Type checking
uv run mypy .

# Build package
uv build
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t cnnparted .

# Run in container
docker run -it cnnparted
```

## 📦 Package Structure

The project is now organized as a modern Python package:

- `pyproject.toml` - Modern Python packaging configuration
- `uv.lock` - Locked dependencies for reproducible builds
- `justfile` - Development task runner
- `install.sh` - Automated installation script
- `dev_setup.sh` - Quick development setup

## 🆕 What's New with uv

### Benefits of the new build system:
- **Faster installations** - uv is 10-100x faster than pip
- **Better dependency resolution** - More reliable and deterministic
- **Locked dependencies** - Reproducible builds across environments
- **Modern tooling** - Uses `pyproject.toml` instead of `setup.py`
- **Simplified workflow** - One-command installation
- **Development-friendly** - Separate dev dependencies and quick setup

### Dependency Management
Dependencies are now managed in `pyproject.toml`:
- **Main dependencies** - Core runtime requirements
- **Dev dependencies** - Testing, formatting, type checking
- **NVIDIA dependencies** - GPU-specific packages

## 🔧 Troubleshooting

### uv not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Research tools not found (accelergy, timeloop, etc.)
If you get errors about missing accelergy or other research tools:
```bash
# These tools are not in PyPI - use the full installation
./install.sh

# Or manually install from submodules with build dependencies
git submodule update --init --recursive
uv pip install pyyaml setuptools wheel  # Build dependencies
uv pip install -e tools/accelergy/ --no-build-isolation
uv pip install -e tools/accelergy-aladdin-plug-in/ --no-build-isolation
uv pip install -e tools/accelergy-cacti-plug-in/ --no-build-isolation
uv pip install -e tools/accelergy-table-based-plug-ins/ --no-build-isolation
```

### Build errors (`ModuleNotFoundError: No module named 'yaml'`)
If you see YAML module errors during installation:
```bash
# Install build dependencies first
uv pip install pyyaml setuptools wheel
# Then re-run the installation
```

### Accelergy issues
If accelergy doesn't run correctly:
```bash
export PATH=$PATH:$(pwd)/cnnparted
accelergyTables
```

### Build failures
If external tools fail to build, ensure you have the required build dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git

# macOS
xcode-select --install
brew install cmake
```

## 📚 Migration from Old Setup

If you have an existing installation, you can migrate to the new system:

```bash
# Remove old virtual environment
rm -rf pythonEnv/

# Run new installation
./install.sh
```

## 🤝 Contributing

1. Set up development environment: `just dev-setup`
2. Install pre-commit hooks: `just setup-hooks`
3. Make your changes
4. Run tests: `just test`
5. Format code: `just format`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use CNNParted in your research, please cite our work:

### Latest Publication (2025)
```bibtex
@inproceedings{kress2025cnnparted,
  author={Kreß, Fabian and Hoefer, Julian and Lin, Qiushi and Schmidt, Patrick and Zhu, Zhenhua and Zhu, Yu and Harbaum, Tanja and Wang, Yu and Becker, Jürgen},
  booktitle={2025 26th International Symposium on Quality Electronic Design (ISQED)},
  title={Deep Neural Network Inference Partitioning in Embedded Hybrid Analog-Digital Systems},
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/ISQED65160.2025.11014471}
}
```

### Original Framework Publication (2023)
```bibtex
@article{kress2023cnnparted,
    title = {{CNNParted}: An open source framework for efficient Convolutional Neural Network inference partitioning in embedded systems},
    author = {Fabian Kreß and Vladimir Sidorenko and Patrick Schmidt and Julian Hoefer and Tim Hotfilter and Iris Walter and Tanja Harbaum and Jürgen Becker},
    journal = {Computer Networks},
    volume = {229},
    pages = {109759},
    year = {2023},
    issn = {1389-1286},
    doi = {https://doi.org/10.1016/j.comnet.2023.109759},
    url = {https://www.sciencedirect.com/science/article/pii/S1389128623002049}
}
```

---

For more information and advanced usage, see the [documentation](docs/) or the examples in the `examples/` directory.
