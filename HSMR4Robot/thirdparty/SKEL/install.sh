sudo apt install python3.12-venv
sudo apt install python3-pip
python3.12 -m venv skel_venv
source skel_venv/bin/activate

pip install -e .
# These are forks with minus fixes to work on the latest Python versions
pip install git+https://github.com/MarilynKeller/chumpy
pip install git+https://github.com/MarilynKeller/aitviewer 

