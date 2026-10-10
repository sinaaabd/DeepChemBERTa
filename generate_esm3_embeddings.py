import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
import pickle
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESM3.from_pretrained("esm3_sm_open_v1")
model.eval()
max_length = 1400
esm3_embeddings = {}
file_path = os.path.join("datasets", "Train.csv")
df = pd.read_csv(file_path)
sequences = df['Sequence'].unique()
i = 1
for sequence in sequences:
    print(i, end=' ')
    i = i + 1
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    with torch.no_grad():
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
    embeddings = output.embeddings   # shape: (L, D)
    embeddings = embeddings.squeeze(0)
    protein_embedding = embeddings.mean(dim=0)

    X = embeddings.squeeze(0).cpu().numpy()  # (L, 1536)

    pca = PCA(n_components=128)
    X_reduced = pca.fit_transform(X)  # (L, 128)
    padded_arr = np.pad(
        X_reduced,
        pad_width=((0, max_length - X_reduced.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0
    )
    esm3_embeddings[sequence] = padded_arr

pickle.dump(esm3_embeddings, open("datasets/ESM3_Embeddings_ChEMBL_PAH_I65T_GCDH_R227P_Residues.pickle", "wb"))