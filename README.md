# CNNParted

## About
CNNParted is a framework for hardware-aware design space exploration of CNN inference partitioning in embedded AI applications. Thereby, not only the hardware deployed in the nodes is considered but also the impact of the link on system energy consumption and inference latency.

The framework currently includes a custom Ethernet model as well as hardware accelerators models taken from Timeloop example repository (Simba, Eyeriss and Simple OS Array).

## Instructions
0. If you have no permission to install python packages on your system, create a virtual environment
    ```sh
    python3.9 -m venv pythonEnv
    source pythonEnv/bin/activate
    pip install --upgrade pip
    pip install wheel nvidia-pyindex pytorch_quantization
    ```

1. Download submodules
    ```sh
    git submodule update --init --recursive
    ```

2. Run installation
    ```sh
    git submodule foreach --recursive 'if [ -f setup.py ]; then pip install .; fi'
    pip install -e .
    pip install -r requirements.txt
    ```

3. Setup the environment using the script provided in env:
    ```sh
    source env/setupEnv.sh
    ```

4. Run the given python script:
    ```sh
    python3.9 cnnparted/cnnparted.py examples/squeezenet1_1.yaml RunName
    ```

## Known issues

If accelergy does not run correctly, add the main directory to `PATH` and execute `accelergyTables`:
```sh
export PATH=$PATH:<path/to/cnn-parted>
accelergyTables
```

## Citing this work

If you found this tool useful, please use the following BibTeX to cite us

```
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


@inproceedings{kress2022hwpart,
    title={Hardware-aware Partitioning of Convolutional Neural Network Inference for Embedded AI Applications},
    author={Kreß, Fabian and Hoefer, Julian and Hotfilter, Tim and Walter, Iris and Sidorenko, Vladimir and Harbaum, Tanja and Becker, Jürgen},
    booktitle={2022 18th International Conference on Distributed Computing in Sensor Systems (DCOSS)},
    pages={133-140},
    year={2022},
    doi={10.1109/DCOSS54816.2022.00034}
}
```
