import subprocess
import os
from framework.constants import ROOT_DIR

class Vampire:
    def __init__(self):
        self.vampire_path = os.path.join(ROOT_DIR,'tools','VAMPIRE')
        self.config_path= os.path.join(self.vampire_path,'configs','default.cfg')
        self.data_dependency_model="MEAN"
        self.parsing_format="ASCII"
        self.spec = os.path.join(self.vampire_path,'dramSpec','example.cfg')

    def run(self, trace_filename,  csv_filename):
        command = [
            "./vampire",
            "-f", trace_filename,
            "-c", self.config_path,
            "-d", self.data_dependency_model,
            "-p", self.parsing_format,
            "-csv",csv_filename,
            "-dramSpec",self.spec
        ]

        try:
            resut = subprocess.run(command,stdout=subprocess.PIPE,cwd=self.vampire_path,check=True )
            return os.path.join(self.vampire_path,csv_filename)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")