conda create --name mllmeval python=3.10
conda activate mllmeval

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt