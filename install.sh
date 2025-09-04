#!/bin/bash

# CNNParted Simplified Installation Script
# Fast and reliable installation with minimal complexity

set -e  # Exit on any error

echo "🚀 Setting up CNNParted..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the CNNParted root directory."
    exit 1
fi

# Function to check and suggest installation of system dependencies
check_and_install_system_dep() {
    local check_cmd="$1"
    local package_name="$2"

    # For library packages, check if they exist in system
    if [[ "$check_cmd" == *"lib"* ]] || [[ "$check_cmd" == *"-dev"* ]]; then
        # Check for development libraries using pkg-config or dpkg
        if ! dpkg -l | grep -q "$check_cmd" 2>/dev/null && ! pkg-config --exists "$check_cmd" 2>/dev/null; then
            echo "❌ Missing system dependency: $check_cmd"
            echo "   Please install with:"
            echo "   Ubuntu/Debian: sudo apt-get install $package_name"
            echo "   CentOS/RHEL: sudo yum install $package_name"
            echo "   macOS: brew install $package_name"
            exit 1
        else
            echo "✅ Found system dependency: $check_cmd"
        fi
    else
        # For command-line tools, check if command exists
        if ! command -v "$check_cmd" >/dev/null 2>&1; then
            echo "❌ Missing system dependency: $check_cmd"
            echo "   Please install with:"
            echo "   Ubuntu/Debian: sudo apt-get install $package_name"
            echo "   CentOS/RHEL: sudo yum install $package_name"
            echo "   macOS: brew install $package_name"
            exit 1
        else
            echo "✅ Found system dependency: $check_cmd"
        fi
    fi
}

# Function to validate Accelergy configuration
validate_accelergy_config() {
    echo "🔧 Validating Accelergy configuration..."

    local config_file="$HOME/.config/accelergy/accelergy_config.yaml"
    if [ ! -f "$config_file" ]; then
        echo "⚠️  Warning: Accelergy config not found at $config_file"
        echo "   Run './install.sh' to regenerate configuration"
        return 1
    fi

    # Check critical component directories exist (based on actual config paths)
    local has_warnings=false
    local venv_primitives="$SIM_ROOT_DIR/.venv/share/accelergy/primitive_component_libs"
    local tools_primitives="$SIM_ROOT_DIR/tools/accelergy/share/primitive_component_libs"
    local venv_estimators="$SIM_ROOT_DIR/.venv/share/accelergy/estimation_plug_ins"
    local tools_estimators="$SIM_ROOT_DIR/tools/accelergy/share/estimation_plug_ins"

    for dir in "$venv_primitives" "$tools_primitives" "$venv_estimators" "$tools_estimators"; do
        if [ ! -d "$dir" ]; then
            if [ "$has_warnings" = false ]; then
                echo "⚠️  Missing Accelergy component directories:"
                has_warnings=true
            fi
            echo "   - $dir"
        fi
    done

    if [ "$has_warnings" = false ]; then
        echo "✅ Accelergy configuration looks good!"
        return 0
    else
        echo "💡 Some Accelergy directories missing, but should still work."
        return 0  # Don't fail installation for missing dirs
    fi
}

# Check for essential dependencies
echo "🔍 Checking essential build dependencies..."

# Basic build tools
for cmd in git make gcc g++ python3; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "❌ Missing required dependency: $cmd"
        echo "   Please install build tools first:"
        echo "   Ubuntu/Debian: sudo apt-get install build-essential git python3 python3-dev"
        echo "   CentOS/RHEL: sudo yum groupinstall 'Development Tools' && sudo yum install python3 python3-devel"
        echo "   macOS: xcode-select --install && brew install python3"
        exit 1
    else
        echo "✅ Found: $cmd"
    fi
done

echo "🔍 Checking additional system dependencies..."

# Additional system dependencies required for timeloop and external tools
check_and_install_system_dep scons "scons"
check_and_install_system_dep clang "clang"
check_and_install_system_dep libconfig++-dev "libconfig++-dev libboost-serialization-dev libyaml-cpp-dev"
check_and_install_system_dep libboost-iostreams-dev "libboost-iostreams-dev libboost-thread-dev"

echo "✅ All required dependencies found"

# Install uv if not available
if ! command -v uv >/dev/null 2>&1; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment and install core dependencies
echo "� Creating virtual environment and installing dependencies..."
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Setup git submodules (required for accelergy tools)
echo "🔄 Setting up git submodules..."
git submodule update --init --recursive

# Install essential accelergy tools
echo "⚙️  Installing accelergy tools..."
uv pip install pyyaml setuptools wheel

# Simple accelergy tool installation
install_tool() {
    local tool_path=$1
    local tool_name=$(basename "$tool_path")

    if [ -d "$tool_path" ]; then
        echo "Installing $tool_name..."
        if [[ "$tool_name" == *"plug-in"* ]]; then
            # Non-editable install for plugins to ensure data files are copied
            uv pip install "$tool_path/" --no-build-isolation
        else
            # Editable install for main accelergy
            uv pip install -e "$tool_path/" --no-build-isolation
        fi
        echo "✅ $tool_name installed"
    else
        echo "⚠️  $tool_path not found - skipping"
    fi
}

# Install core accelergy tools
install_tool "tools/accelergy"
install_tool "tools/accelergy-aladdin-plug-in"
install_tool "tools/accelergy-cacti-plug-in"
install_tool "tools/accelergy-table-based-plug-ins"

# Configure Accelergy paths
echo "🔧 Configuring Accelergy..."
mkdir -p ~/.config/accelergy
cat > ~/.config/accelergy/accelergy_config.yaml << EOF
version: '0.4'
estimator_plug_ins:
  - ${PWD}/.venv/share/accelergy/estimation_plug_ins
  - ${PWD}/tools/accelergy/share/estimation_plug_ins
primitive_components:
  - ${PWD}/.venv/share/accelergy/primitive_component_libs
  - ${PWD}/tools/accelergy/share/primitive_component_libs
compound_components: []
EOF

# Validate Accelergy configuration
validate_accelergy_config

# Install model explorer
if [ -d "tools/dnn-model-exploration" ]; then
    echo "🧠 Installing model explorer..."
    cd tools/dnn-model-exploration
    uv pip install -e . --no-build-isolation
    cd ../..
    echo "✅ Model explorer installed"
fi

# Build essential external tools
echo "🔧 Building external tools..."
export SIM_ROOT_DIR=$PWD

# Build CACTI (simple make)
if [ -d "tools/cacti" ]; then
    echo "Building CACTI..."
    make -C tools/cacti
    echo "✅ CACTI built"
fi

# Setup Timeloop
if [ -d "tools/timeloop" ]; then
    echo "🕒 Setting up Timeloop..."

    # Create required symlink for PAT (Power Area Timing) model
    if [ ! -d "tools/timeloop/src/pat" ]; then
        echo "Creating PAT symlink..."
        ln -sf ../pat-public/src/pat tools/timeloop/src/pat
        echo "✅ PAT symlink created"
    fi

    # Fix missing header in compound-config.hpp for C++17 compatibility
    COMPOUND_CONFIG_FILE="tools/timeloop/include/compound-config/compound-config.hpp"
    if [ -f "$COMPOUND_CONFIG_FILE" ] && ! grep -q "#include <cstdint>" "$COMPOUND_CONFIG_FILE"; then
        echo "Fixing compound-config.hpp header..."
        sed -i '/#pragma once/a #include <cstdint>' "$COMPOUND_CONFIG_FILE"
        echo "✅ Header fix applied"
    fi

    # Build Timeloop with scons
    echo "Building Timeloop (this may take several minutes)..."
    scons -C tools/timeloop -j4 --accelergy

    # Verify build success
    if [ -f "tools/timeloop/bin/timeloop-mapper" ]; then
        echo "✅ Timeloop built successfully"
        TIMELOOP_SIZE=$(du -h tools/timeloop/bin/timeloop-mapper | cut -f1)
        echo "   timeloop-mapper: $TIMELOOP_SIZE"
    else
        echo "❌ Timeloop build failed - timeloop-mapper not found"
        exit 1
    fi

    # Fix Timeloop configuration files for proper YAML syntax
    echo "🔧 Fixing Timeloop configuration files..."

    # Fix unquoted string literals that cause Accelergy syntax errors
    if [ -d "configs/tl_configs" ]; then
        # Fix technology field (45nm -> "45nm")
        find configs/tl_configs -name "*.yaml" -exec grep -l "technology: 45nm" {} \; | xargs -r sed -i 's/technology: 45nm/technology: "45nm"/g'

        # Fix memory type field (LPDDR4 -> "LPDDR4")
        find configs/tl_configs -name "*.yaml" -exec grep -l "type: LPDDR4" {} \; | xargs -r sed -i 's/type: LPDDR4/type: "LPDDR4"/g'

        echo "✅ Configuration files fixed"
    fi
else
    echo "⚠️  Timeloop directory not found - skipping"
fi

# Create comprehensive environment activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
export SIM_ROOT_DIR=$PWD
source .venv/bin/activate

# Add Timeloop library path for shared libraries
export LD_LIBRARY_PATH=$PWD/tools/timeloop/lib:${LD_LIBRARY_PATH}

# Add model explorer and current directory to Python path
export PYTHONPATH=$PWD/tools/dnn-model-exploration:$PWD:${PYTHONPATH}

# Add external tools to PATH
export PATH="${SIM_ROOT_DIR}/tools/cacti:$PATH"
export PATH="${SIM_ROOT_DIR}/tools/timeloop/bin:$PATH"

# Accelergy validation function
validate_accelergy_config() {
    echo "🔧 Validating Accelergy configuration..."

    local config_file="$HOME/.config/accelergy/accelergy_config.yaml"
    if [ ! -f "$config_file" ]; then
        echo "⚠️  Warning: Accelergy config not found at $config_file"
        echo "   Run './install.sh' to regenerate configuration"
        return 1
    fi

    # Check critical component directories exist (based on actual config paths)
    local has_warnings=false
    local venv_primitives="$SIM_ROOT_DIR/.venv/share/accelergy/primitive_component_libs"
    local tools_primitives="$SIM_ROOT_DIR/tools/accelergy/share/primitive_component_libs"
    local venv_estimators="$SIM_ROOT_DIR/.venv/share/accelergy/estimation_plug_ins"
    local tools_estimators="$SIM_ROOT_DIR/tools/accelergy/share/estimation_plug_ins"

    for dir in "$venv_primitives" "$tools_primitives" "$venv_estimators" "$tools_estimators"; do
        if [ ! -d "$dir" ]; then
            if [ "$has_warnings" = false ]; then
                echo "⚠️  Missing Accelergy component directories:"
                has_warnings=true
            fi
            echo "   - $dir"
        fi
    done

    if [ "$has_warnings" = false ]; then
        echo "✅ Accelergy configuration looks good!"
        return 0
    else
        echo "💡 Some Accelergy directories missing, but should still work."
        return 0  # Don't fail installation for missing dirs
    fi
}

# Run validation
validate_accelergy_config

echo "✅ CNNParted environment activated!"
echo "📍 SIM_ROOT_DIR: $SIM_ROOT_DIR"
echo "🐍 Python environment: $(which python)"
echo "🎯 Ready to use: cnnparted examples/squeezenet1_1.yaml test_run"
EOF

chmod +x activate_env.sh

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source activate_env.sh"
echo ""
echo "To test the installation:"
echo "  cnnparted examples/squeezenet1_1.yaml test_run"
