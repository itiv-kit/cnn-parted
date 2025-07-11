from setuptools import setup
from setuptools.command.install import install
import os

class ToolsInstall(install):
    def run(self):
        install.run(self)
        path = os.getcwd().replace(" ", "\ ").replace("(","\(").replace(")","\)") + "/tools/"
        os.system("echo 'Setting up CNNParted Tools'")

        # Install CACTI
        os.system("make -C"+path+"cacti")
        os.system("chmod -R 777 "+path+"cacti")

        # Install Ramulator and vampire
        os.system("echo 'Setting up ramulator and vampire")

        os.system("make -j -C"+path+"ramulator")
        os.system("make -j -C"+path+"VAMPIRE")

        # Install Timeloop
        os.system("chmod +x "+path+"setupTimeloop.sh")
        os.system("sh "+path+"setupTimeloop.sh")

        # Update MNSIM
        os.system("git -C "+path+"MNSIM-2.0 apply ../mnsim.patch")



setup(
    name='CNNParted',
      version='1.0',
      description='CNN Inference Partitioning Framework',
      url='https://github.com/itiv-kit/cnn-parted',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3'
      ],
      keywords='cnn inference hardware accelerator',
      author='Fabian Kreß',
      author_email='fabian.kress@kit.edu',
      license='MIT',
      packages=['cnnparted'],
      install_requires = [
            'dataclasses',
            'libconf',
            'numpy',
            'Pillow',
            'pyfiglet',
            'PyYAML',
            'scipy',
            'torch>=1.13.1',
            'torchinfo',
            'torchvision',
            'typing_extensions',
            'yamlordereddictloader',
            'tqdm',
            'accelergy',
            'accelergy-aladdin-plug-in',
            'accelergy-cacti-plug-in',
            'accelergy-table-based-plug-ins',
            'model_explorer',
            'onnx',
            'networkx',
            'onnxruntime-gpu',
            'traitlets',
            'pymoo',
            'onnx2pytorch',
            'pandas',
            'joblib',
            'onnx2pytorch',
            'ruamel.yaml',
            "plotly",
            ],
      python_requires = '>=3.11',
      include_package_data = True,
      scripts=['tools/setupTimeloop.sh'],
      cmdclass={'install': ToolsInstall}
    )
