import subprocess
import os
# from app import *
current_location = os.getcwd()
lung_model_path = os.path.join(current_location, 'requirements.txt')

subprocess.run(['pip', 'install', '-r', lung_model_path], shell=True) # install
subprocess.run('streamlit run app.py', shell=True) # run app.py (streamlit app)