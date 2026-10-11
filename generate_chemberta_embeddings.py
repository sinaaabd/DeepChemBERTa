from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import os


def cxsmiles_to_canonical(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def save_embeddings(embeddings, output_path):
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(embeddings, f)
    os.replace(tmp_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file_adr_name", required=True)
    parser.add_argument("--smiles_csv_header", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--convert_to_canonical", action="store_true")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV if you convert to Canonical SMILES; defaults to not overwriting the input"
    )

    args = parser.parse_args()

    CT_df = pd.read_csv(args.csv_file)

    if args.smiles_csv_header not in CT_df.columns:
        raise ValueError(
            f"Column '{args.smiles_csv_header}' not found in CSV."
        )

    if args.convert_to_canonical:
        CT_df["Canonical SMILES"] = CT_df[
            args.smiles_csv_header
        ].apply(cxsmiles_to_canonical)
    elif args.smiles_csv_header != "Canonical SMILES":
        CT_df = CT_df.rename(
            columns={args.smiles_csv_header: "Canonical SMILES"}
        )

    if args.output_csv:
        CT_df.to_csv(args.output_csv, index=False)

    # Remove missing and invalid SMILES.
    valid_smiles = CT_df["Canonical SMILES"].dropna()
    valid_smiles = valid_smiles[
        valid_smiles.map(lambda s: isinstance(s, str) and bool(s.strip()))
    ]
    all_smiles = valid_smiles.unique()

    model_name = "DeepChem/ChemBERTa-77M-MLM"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to("cpu")

    output_path = args.pickle_file_adr_name
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            chemberta_embeddings = pickle.load(f)
    else:
        chemberta_embeddings = {}

    for i, smiles in enumerate(all_smiles, start=1):
        if smiles in chemberta_embeddings:
            continue

        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.inference_mode():
            outputs = model(**inputs)

        # Option A: first-token (CLS-style) embedding.
        #embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        # Option B: mean pooling over non-padding tokens.
        mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        embedding = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )[0].cpu().numpy()

        chemberta_embeddings[smiles] = embedding.astype(np.float32)

        if i % 1000 == 0:
            save_embeddings(chemberta_embeddings, output_path)
            print(f"Processed {i}/{len(all_smiles)} unique SMILES")

    save_embeddings(chemberta_embeddings, output_path)

    print(f"Saved {len(chemberta_embeddings)} embeddings to {output_path}")