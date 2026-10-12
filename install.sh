export UV_EXTRA_INDEX_URL="https://pypi.nvidia.com"

export ROOT_DIR=$(pwd)
export TIMELOOP_INCLUDE_PATH=$ROOT_DIR/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/include
export TIMELOOP_LIB_PATH=$ROOT_DIR/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/lib

# Install Python dependencies with uv and from submodules
echo "--- Starting UV Sync ---"
uv sync --verbose
mkdir .chrome/
uv run plotly_get_chrome -y --path "$ROOT_DIR/.chrome"
echo "--- UV Sync Complete ---"
source env/setupEnv.sh 

echo "Syncing git submodules"
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule sync --recursive
git submodule update --init --recursive

echo "Installing Timeloop and Accelergy"
## Patch Timeloop sconscript
TIMELOOP_SCONSTRUCT="cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/SConstruct"
# Inject the flag to ignore the ISL header warnings
# We use 'a' (append) to add the line after the Environment initialization
sed -i "/env = Environment(ENV = os.environ)/a \env.Append(CXXFLAGS=['-Wno-missing-field-initializers'])" "$TIMELOOP_SCONSTRUCT"

source cnnparted/tools/install_tools.sh
make -C cnnparted/tools local_install_accelergy
make -C cnnparted/tools local_install_timeloop

echo "Running installation of submodules"

# Install from setup.py file,
git submodule foreach --recursive '
    if [ -f setup.py ] && [[ $name =~ "accelergy" ]]; then 
        uv pip install --no-build-isolation .; 
    elif [ -f setup.py ]; then 
        uv pip install --no-build-isolation -e .; 
    fi
'
make -C cnnparted/tools local_install_timeloop_python

# Install from requirements.txt
git submodule foreach --recursive '
    if [ -f requirements.txt ]; then 
        uv pip install --no-build-isolation -r requirements.txt
    fi
'

uv pip install -e .
