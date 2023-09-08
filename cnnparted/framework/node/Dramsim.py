# import subprocess
# import os
# from framework.constants import ROOT_DIR


# class Dramsim:
#     def __init__(self):
#         self.dramsim_path = os.path.join(ROOT_DIR, 'tools','DRAMsim3')


#     # # help
#     # ./build/dramsim3main -h

#     # # Running random stream with a config file
#     # ./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini --stream random -c 100000 

#     # # Running a trace file
#     # ./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini -c 100000 -t sample_trace.txt

#     def run(self,ini_file,cyles,trace):
#         try:
#             current_dir = os.getcwd()
#             os.chdir(self.dramsim_path)

#             command = ["./build/dramsim3main", ini_file, "-c", str(cyles), "-t", trace]
#             subprocess.run(command, check=True)
#             print("Command executed successfully.")
#         except subprocess.CalledProcessError as e:
#             print(f"An error occurred while running the command:{command}, {e}")
#         finally:
#             os.chdir(current_dir)