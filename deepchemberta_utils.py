'''
DeepChemBERTa: Compound-based Protein Stabilization Prediction
Author: Dr. Sina Abdollahi
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Activation
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from DeepChemBERTa import Drug_Target_Prediction
import tensorflow as tf
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import pickle
import time
import os


#ChemBERTa embeddings: A dictionary with canonical SMILES as keys, and ChemBERTa embeddings as values
chemberta_embeddings_dic = pickle.load(open('Compounds_ChemBERTa_embeddings.pickle', 'rb'))
#A dictionary with Sequences as keys, and numpy arrays with shape of (1400, 128) as values representing the ESM3 of the residues
esm3_embeddings_dic = pickle.load(open('ESM3_Embeddings_ChEMBL_PAH_I65T_GCDH_R227P_Residues.pickle', 'rb'))



def generate_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if(mol is None):
        raise ValueError(f"Invalid SMILES string: {smiles}")
    generator = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = np.array(generator.GetFingerprintAsNumPy(mol))
    return fp


def get_ChemBERTa_embeddings(smiles):
    if(smiles not in chemberta_embeddings_dic):
        raise ValueError(f"There is no extracted ChemBERTa embeddings: {smiles}")
    return chemberta_embeddings_dic[smiles]


def get_ESM3_embeddings(sequence):
    if(sequence not in esm3_embeddings_dic):
        raise ValueError(f"There is no extracted ESM3 embeddings: {sequence}")
    return esm3_embeddings_dic[sequence]


def parse_to_standard(data_dir, with_label=True):
    data = pd.read_csv(data_dir)
    # get maximum protein length
    tsa_col = "delta_Thermal_Shift"
    
    # START: Rescale the values Higher than 6.0 and less than -6.0: For Training; Validation; Test Sets
    low_mask = data[tsa_col] < -6.0
    high_mask = data[tsa_col] > 6.0
    # min/max for scaling
    low_min = data.loc[low_mask, tsa_col].min()
    low_max = -6.0
    high_min = 6.0
    high_max = data.loc[high_mask, tsa_col].max()
    # scale low values to [-7, -6]
    data.loc[low_mask, tsa_col] = -7 + (data.loc[low_mask, tsa_col] - low_min) * (-6 + 7) / (low_max - low_min)
    # scale high values to [6, 7]
    data.loc[high_mask, tsa_col] = 6 + (data.loc[high_mask, tsa_col] - high_min) * (7 - 6) / (high_max - high_min)
    # END: Rescale the values Higher than 6.0 and less than -6.0
    
    protein_feature = np.stack(
        [get_ESM3_embeddings(seq) for seq in  data["Sequence"]]
    )
    drug_feature = np.stack([generate_morgan_fp(s) for s in data["Canonical SMILES"]])
    chemberta_features = np.stack(
        [get_ChemBERTa_embeddings(smiles) for smiles in  data["Canonical SMILES"]]
    )
    if(with_label):        
        bin_edges = [-np.inf,-6,-4.5,-3.5,-2.5,-1.5,-1.0,-0.8,-0.6,-0.24,0.0,0.5,1.0,1.5,2.0,2.5,3.2,4,4.8,6.0,np.inf]
        data['bin'] = pd.cut(
            data['delta_Thermal_Shift'],
            bins=bin_edges,
            right=False   # Left inclusive, Right exclusive
        )
        bin_counts = data['bin'].value_counts().sort_index()
        alpha = 0.35
        weights_per_bin = 1.0 / np.power(bin_counts, alpha)
        data['weight'] = data['bin'].map(weights_per_bin).astype(float)
        # Normalize
        data['weight'] /= data['weight'].mean()
        therm_values = data[tsa_col].values
        weights = data['weight'].values
        
        return_dic = {"protein_feature": protein_feature, "drug_feature": drug_feature, "chemberta_features": chemberta_features, "therm_values": therm_values, "sample_weight": weights}
        return return_dic
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, "chemberta_features": chemberta_features}



def parse_for_inference(data_dir):
    data = pd.read_csv(data_dir)
    # Get the maximum protein length
    
    protein_feature = np.stack(
        [get_ESM3_embeddings(seq) for seq in  data["Sequence"]]
    )
    drug_feature = np.stack([generate_morgan_fp(s) for s in data["Canonical SMILES"]])
    chemberta_features = np.stack(
        [get_ChemBERTa_embeddings(smiles) for smiles in  data["Canonical SMILES"]]
    )
    return {"protein_feature": protein_feature, "drug_feature": drug_feature, "chemberta_features": chemberta_features}, data['Sequence'].to_numpy(), data['Canonical SMILES'].to_numpy() 
        


def calc_performance_metrics(test_label, prediction):
    fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
    AUC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(test_label,prediction)
    distance = (1-fpr)**2+(1-tpr)**2
    EERs = (1-recall)/(1-precision)
    positive = sum(test_label)
    negative = test_label.shape[0]-positive
    ratio = negative/positive
    opt_t_AUC = thresholds_AUC[np.argmin(distance)]
    opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
    AUPR = auc(recall,precision)
    return AUC, AUPR, opt_t_AUC, opt_t_AUPR


def get_args():
    args = argparse.Namespace(
        test_name="test",
        test_dti_dir=None,
        test_drug_dir=None, 
        test_protein_dir=None,
        with_label=True,
        window_sizes=[15, 20, 25, 30],
        protein_layers=[128],
        drug_layers=[512, 128],
        fc_layers=[384, 128],
        final_fc_layers=[32, 8],
        learning_rate= 1e-4, 
        n_epoch=150,
        prot_vec="Convolution",
        prot_len=1400,
        aa_dim=128,
        drug_vec="morgan_fp_r2",
        drug_len=2048,
        activation="relu",
        dropout=0.2,
        n_filters=128,
        batch_size=128,
        decay=1e-3,
        validation=False,
        predict=False,
        save_model=None,
        output=None,
    )
    return args



def run_deepchemberta(data_path, result_path=None, load_modell=None, save_model=None, train=None, validation=None, test=None, inference=None):
    if(result_path is not None):
        result_base_path = result_path
    args = get_args()

    drug_layers = args.drug_layers
    window_sizes = args.window_sizes
    if(window_sizes == 0):
        window_sizes = None
    protein_layers = args.protein_layers
    fc_layers = args.fc_layers
    # Training parameters
    train_params = {
        "batch_size": args.batch_size,
    }
    # Type parameters
    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "aa_dim": args.aa_dim,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    # Model parameters
    model_params = {
        "drug_layers": drug_layers,
        "protein_windows": window_sizes,
        "protein_layers": protein_layers,
        "fc_layers": fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout,
    }

    dti_prediction_model = ""
    model_params.update(type_params)
    print("\tModel parameters summary\t")
    print("=====================================================")
    for key in model_params.keys():
        print("{:20s} : {:10s}".format(key, str(model_params[key])))
    print("=====================================================")
    dti_prediction_model = Drug_Target_Prediction(**model_params)
    # For instance: my_model.h5
    if(load_modell):
        print(f">>>>>>>>>>>>>>>>> Loading the model (DeepChemBERTa: Compound-based Stabilization Predictor): {load_modell}")
        dti_prediction_model.load(f'saved_models/{load_modell}')
    #dti_prediction_model.summary()
    
    #########################################Training Section#########################################
    if(train):
        print(f">>>>>>>>>>>>>>>>> Training the model using {train} file!")
        data_path_train = os.path.join(data_path, f'{train}.csv')
        train_dic = parse_to_standard(data_path_train, with_label=True)
        
        train_dic = {"protein_feature": train_dic["protein_feature"], "drug_feature": train_dic["drug_feature"], "chemberta_features":  train_dic["chemberta_features"], "therm_values": train_dic["therm_values"], "sample_weight": train_dic["sample_weight"]}
        
        train_dic.update(train_params)
        print(train_dic['protein_feature'].shape)
        print(train_dic['chemberta_features'].shape)
        print(train_dic['drug_feature'].shape)
        # Training Time
        train_start = time.time()
        if(validation):
            best_rmse = 100
            best_weights = None
            for epoch in range(args.n_epoch):
                print(f"\nEpoch {epoch + 1}/{args.n_epoch}")
                trained_model = dti_prediction_model.fit(**train_dic)

                data_path_val = os.path.join(data_path, f'{validation}.csv')
                val_dic = parse_to_standard(data_path_val, with_label=True)
                
                val_dic = {"test": val_dic}
                current_rmse = test_tsa_model(dti_prediction_model, val_dic, result_path, validation, True)
                print(f'Epoch RMSE: {current_rmse}')
                if(current_rmse < best_rmse):
                    best_rmse = current_rmse
                    best_weights = trained_model.get_weights()
            dti_prediction_model.update_model_weights(best_weights)
        else:
            for epoch in range(args.n_epoch):
                print(f"\nEpoch {epoch + 1}/{args.n_epoch}")
                dti_prediction_model.fit(**train_dic)
        if(validation):        
            pd.DataFrame(dti_prediction_model.training_losses).to_csv(f"{result_path}/Training_Validation_Loss.csv", index=False)
        else:
            dd = {}
            dd["Training Loss"] = dti_prediction_model.training_losses["Training Loss"]
            pd.DataFrame(dd).to_csv(f"{result_path}/Training_Loss.csv", index=False)
        
        train_end = time.time()
        print(f"\tTraining time: {train_end - train_start}")
    
    if(save_model):
        print(f"Saving the model: {save_model}")
        dti_prediction_model.save(f'saved_models/{save_model}')
    result_df = ""
    #########################################Testing Section#########################################
    if(test):
        print(f">>>>>>>>>>>>>>>>> Testing the model using {test} file!")
        data_path_test = os.path.join(data_path, f'{test}.csv')
        inference_start = time.time()
        test_dic = parse_to_standard(data_path_test, with_label=True)
        test_dic = {"test": test_dic}
        rmse = test_tsa_model(dti_prediction_model, test_dic, result_path, test, False)
        print(f"MSE = {rmse}")
    #########################################Inference Section#######################################
    if(inference):
        print(f">>>>>>>>>>>>>>>>> Inference for {inference} file!")
        data_path_test = os.path.join(data_path, f'{inference}.csv')
        inference_start = time.time()
        parsed_data, seqs, cpds = parse_for_inference(data_path_test)
        test_dic = {"test": parsed_data}
        test_predicted = dti_prediction_model.inference(**test_dic)
        inference_end = time.time()
        print(f"\tInference time: {inference_end - inference_start}")
        test_predicted = test_predicted["test"]
        prediction = test_predicted["predicted"]
        value = np.squeeze(prediction)
        result_df = pd.DataFrame()
        result_df["Sequence"] = seqs
        result_df["Canonical SMILES"] = cpds
        result_df["prediction"] = value
        name_of_result = inference.split('.csv')[0]
        if(result_path is not None):
            result_df.to_csv(f"{result_base_path}/inference_{name_of_result}.csv", index=False)
            
    return result_df

def test_tsa_model(model, test_dic, result_path, test, is_validation=False):
    inference_start = time.time()
    test_predicted = model.predict(**test_dic)
    inference_end = time.time()

    test_predicted = test_predicted["test"]
    therm_pred = np.squeeze(test_predicted["predicted"])
    therm_true = np.squeeze(test_predicted["therm_true"])
    
    y_pred_bin = (therm_pred >= 0.5).astype(int)
    y_true_bin = (therm_true >= 0.5).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    print(f"Therm   → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Evaluate full model with all outputs
    test_loss = model._Drug_Target_Prediction__model_t.evaluate(
        x = [test_dic["test"]["chemberta_features"],
           test_dic["test"]["drug_feature"],
           test_dic["test"]["protein_feature"]],
        y = therm_true,
        verbose=0
    )

    if is_validation:
        model.training_losses["Validation Loss"].append(test_loss[0])

    print(f'Validation/Test Loss={test_loss[0]}')
    for name, val in zip(model._Drug_Target_Prediction__model_t.metrics_names, test_loss):
        print(f"{name}: {val}")

    # Regression Metrics
    therm_rmse = np.sqrt(mean_squared_error(therm_true, therm_pred))
    therm_mae = mean_absolute_error(therm_true, therm_pred)
    therm_r2 = r2_score(therm_true, therm_pred)

    print(f"Therm   → RMSE: {therm_rmse:.4f}, MAE: {therm_mae:.4f}, R²: {therm_r2:.4f}")

    # Save results
    name_of_result = test.split('.csv')[0]
    result_df = pd.DataFrame({
        "predicted": therm_pred,
        "therm_true": therm_true
    })

    if not is_validation:
        print(f"\tInference time: {inference_end - inference_start:.2f} sec")
        if result_path is not None:
            result_df.to_csv(f"{result_path}/prediction_{name_of_result}.csv", index=False)

        metrics_df = pd.DataFrame({
            "Therm_RMSE": [therm_rmse],
            "Therm_MAE": [therm_mae],
            "Therm_R2": [therm_r2]
        })
        metrics_df.to_csv(f"{result_path}/Result_{name_of_result}.csv", index=False)
    mask = (therm_true > 0.5) | (therm_true < -0.8)
    filtered_true = therm_true[mask]
    filtered_pred = therm_pred[mask]
    filtered_rmse = 100
    print(f"Filtered samples: {np.sum(mask)} / {len(therm_true)}")
    if len(filtered_true) > 0:
        filtered_rmse = np.sqrt(mean_squared_error(filtered_true, filtered_pred))
    else:
        filtered_rmse = np.nan  # Avoid crash if no samples

    print(f"Filtered RMSE (therm_true > 0.5 or < -0.8): {filtered_rmse:.4f}")
    return filtered_rmse
