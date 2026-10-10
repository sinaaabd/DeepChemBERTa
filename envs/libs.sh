conda install -c conda-forge \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    jupyter \
    rdkit \
    -y


pip install tensorflow==2.15.*

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install "transformers>=4.30"

pip install sentencepiece datasets accelerate
