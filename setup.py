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

        # Install Timeloop
        os.system("chmod +x "+path+"setupTimeloop.sh")
        os.system("sh "+path+"setupTimeloop.sh")

        # Install DRAMsim3 
        os.system("chmod +x " + path + "setupDramsim3.sh")
        os.system("sh " + path + "setupDramsim3.sh") # CMake is needed

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
            'dataclasses==0.8',
            'libconf==2.0.1',
            'numpy==1.19.5',
            'Pillow==8.4.0',
            'pyfiglet==0.8.post1',
            'PyYAML==6.0',
            'scipy==1.5.4',
            'torch==1.10.1',
            'torchinfo==1.5.4',
            'torchvision==0.11.2',
            'typing_extensions==4.1.1',
            'yamlordereddictloader==0.4.0',
            'accelergy==0.3',
            'accelergy-aladdin-plug-in==0.1',
            'accelergy-cacti-plug-in==0.1',
            'accelergy-table-based-plug-ins==0.1',
            ],
      python_requires = '>=3.6',
      include_package_data = True,
      scripts=['tools/setupTimeloop.sh'],
      cmdclass={'install': ToolsInstall},
    )
