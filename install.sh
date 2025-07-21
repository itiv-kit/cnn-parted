git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule sync --recursive
git submodule update --init --recursive

# Install Python dependencies with uv and from submodules
uv sync
source env/setupEnv.sh 
uv pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization
git -C cnnparted/tools/accelergy-table-based-plug-ins/ apply  ../accelergy_table_plugins.patch
git submodule foreach --recursive 'if [ -f setup.py ] && [[ $name =~ "accelergy" ]]; then uv pip install .; elif [ -f setup.py ]; then uv pip install -e .; fi'
git submodule foreach --recursive 'if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi'


# Apply patches to MNSIM, install Cacti and Timeloop
source cnnparted/tools/install_tools.sh
#sudo ln -s /usr/lib/libstdc++.so.6 /usr/lib/libstdc++.so #This step might be needed for Timeloop installation
source cnnparted/tools/setupTimeloop.sh
uv pip install -e .