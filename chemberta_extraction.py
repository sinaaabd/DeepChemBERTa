'''
DeepChemBERTa: Compound-based Protein Stabilization Prediction
Author: Dr. Sina Abdollahi
'''
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import time
import sys
import os



# Convert Cxsmiles to Canonical SMILES
def cxsmiles_to_canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file_adr_name', required=True, type=str, help='ChemBERTa Embeddings Pickle File')
    parser.add_argument('--smiles_csv_header', required=True, type=str, help='SMILES header in the CSV file')
    parser.add_argument('--csv_file', required=True, type=str, help='The CSV file containing Compounds or Compound-Targets information')
    parser.add_argument('--convert_to_canonical', help='Do you want to convert the SMILES notation to its Canonical version?', action="store_true")
   
    args = parser.parse_args()

    CT_df = pd.read_csv(args.csv_file)
    if(args.convert_to_canonical):
        print("Conversion to Canonical SMILES!")
        CT_df[f'{args.smiles_csv_header}_tmp'] = CT_df[args.smiles_csv_header].apply(cxsmiles_to_canonical)
        CT_df = CT_df.drop(columns=[args.smiles_csv_header])
        CT_df = CT_df.rename(columns={f'{args.smiles_csv_header}_tmp': 'Canonical SMILES'})
    else:
        CT_df = CT_df.rename(columns={args.smiles_csv_header: 'Canonical SMILES'})
    CT_df.to_csv(args.csv_file, index=False)

    all_smiles = CT_df['Canonical SMILES'].unique()

    model_name = "DeepChem/ChemBERTa-77M-MLM"
    # Load tokenizer and model (no classification pooler needed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval().to("cpu")  # or "cuda" if available

    chemberta_embeddings = {}
    if(os.path.exists(args.pickle_file_adr_name)):
        chemberta_embeddings = pickle.load(open(args.pickle_file_adr_name,'rb'))
    i = 0
    j = 1
    for smiles in all_smiles:
        if(smiles not in chemberta_embeddings):
            i += 1
            if(i % 1000 == 0):
                print(j, end='\t')
                j += 1
                pickle.dump(chemberta_embeddings, open(args.pickle_file_adr_name, 'wb'))
            # Tokenize the SMILES
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            # Get embedding (CLS token)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
            # Convert to numpy array
            embedding = cls_embedding.squeeze().cpu().numpy()  # shape: (hidden_dim,)
            chemberta_embeddings[smiles] = embedding

    pickle.dump(chemberta_embeddings, open(args.pickle_file_adr_name, 'wb'))
