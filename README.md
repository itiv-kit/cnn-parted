# CNNParted

## About
CNNParted is a framework for hardware-aware design space exploration of CNN inference partitioning in embedded AI applications. Thereby, not only the hardware deployed in the nodes is considered but also the impact of the link on system energy consumption and inference latency.

The framework currently includes a custom Ethernet model as well as hardware accelerators models taken from Timeloop example repository (Simba, Eyeriss and Gemmini).
Additionally, we support evaluating accelerators with ZigZag.

## Installation Instructions
We provide an installation script with all necessary steps. To simplify the installation of all dependencies, the usage of a Docker Container is highly recommended. A devcontainer file for use with VSCode is also provided. Otherwise, Timeloop dependencies might need to be manually installed.
To manage the dependencies for the Python module, we use uv.
    ```sh
    ./install.sh
    ```

A more detailed breakdown of the installation changes
0. First, ensure all submodules can be cloned with HTTPS instead of SSH and clone them:
    ```sh
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    git submodule sync --recursive
    git submodule update --init --recursive
    ```

1. Install all Python dependencies with uv and from the submodules:
    ```sh
    uv sync
    source env/setupEnv.sh 
    uv pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization
    git -C cnnparted/tools/accelergy-table-based-plug-ins/ apply  ../accelergy_table_plugins.patch
    git submodule foreach --recursive 'if [ -f setup.py ] && [[ $name =~ "accelergy" ]]; then uv pip install .; elif [ -f setup.py ]; then uv pip install -e .; fi'
    git submodule foreach --recursive 'if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi'
    ```

2. Install remaining tools, apply patches for compatibility:
    ```sh
    source cnnparted/tools/install_tools.sh
    source cnnparted/tools/setupTimeloop.sh
    ```


3. Run the given python script:
    ```sh
    # If not already in virtual environment, activate it:
    source env/setupEnv.sh
    python3.11 cnnparted/cnnparted.py examples/squeezenet1_1.yaml RunName
    ```

## Known issues

If accelergy does not run correctly, add the main directory to `PATH` and execute `accelergyTables`:
```sh
export PATH=$PATH:<path/to/cnn-parted/cnnparted>
accelergyTables
```

## Citing this work

If you found this tool useful, please use the following BibTeX to cite us

```
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
