# DeepChemBERTa_Satbilization

## DeepChemBERTa Architecture
DeepChemBERTa_Stabilization utilizes trainable embeddings to encode amino acids of proteins and utilizes Morgan drug fingerprint and ChemBERTa embeddings to capture the features of compounds. It uses several convolutional windows with different sizes to capture both local and global features of protein. DeepChemBERTa utilizes a Power-law weighting system with alpha=0.35. Besides, it utilizes batch variance penalty and Huber (delta=2.0) as a loss function, where batch size should be more than 128.
DeepChemBERTa_Stabilization predicts thermal shift assay values.

### Prerequisite
To run DeepChemBERTa_Stabilization, first install the dependencies using conda:
```
conda create -n deepchemberta_tf python=3.10 -y
conda activate deepchemberta_tf

conda install -c conda-forge numpy pandas scikit-learn tqdm jupyter rdkit -y

pip install tensorflow[and-cuda]==2.15.*
```

Before using the model, make sure that you have already extracted the ChemBERTa features for the compounds used in the training, test, validation, or inference sets, and stored them in a pickle file. Moreover, ensure that the compound representations are converted into Canonical SMILES format. You can perform both the conversion and the feature extraction using the following command:

```
python generate_chemberta_embeddings.py --smiles_csv_header <smiles-col-header> --csv_file <input-data-complete-name> --pickle_file_adr_name Compounds_ChemBERTa_embeddings.pickle --output_csv <output-csv-file-containing-canonical-SMILES> --convert_to_canonical 
```

If you do not need the conversion, simply remove the --convert_to_canonical option. Also, make sure that you have installed the following packages on the CPU beforehand:
```
pip install transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Datasets
The datasets must be in CSV format and contain at least three columns with the headers: "Canonical SMILES", "Sequence", and "delta_Thermal_Shift", separated by commas. Each dataset should include training, validation, and test sets. For each run, create a dedicated folder within the datasets/ directory and place the corresponding files in that folder.

## Execute on TSA data
To train DeepChemBERTa_Stabilization from scratch using a TSA dataset and save the trained model, use the following command:
```
python main.py --data <data-dir> --train <training-file-without.csv> --test <test-file-without.csv> --val <validation-file-without.csv> --save_model <name-of-trained-model>
```
The above command, after training the model, it saves the trained model in saved_models/ folder.
