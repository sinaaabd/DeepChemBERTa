'''
DeepChemBERTa: Compound-based Protein Stabilization Prediction
Author: Dr. Sina Abdollahi
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation, Dropout,
    Conv1D, GlobalMaxPooling1D, Embedding,
    SpatialDropout1D, Concatenate, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_recall_curve, auc, roc_curve


def variance_loss(lambda_var=0.1, delta=2.0):
    huber = tf.keras.losses.Huber(delta=delta,
                                  reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        huber_loss = tf.reduce_mean(huber(y_true, y_pred))
        var_true = tf.math.reduce_variance(y_true)
        var_pred = tf.math.reduce_variance(y_pred)
        var_penalty = tf.square((var_pred - var_true) / (var_true + 1e-6))
        return huber_loss + lambda_var * var_penalty
    return loss


class Drug_Target_Prediction(object):

    def __init__(self, dropout=0.2, drug_layers=(512, 128), protein_windows=(10, 15, 20, 25, 30),
                 filters=128, learning_rate=1e-4, decay=1e-4, fc_layers=None, final_fc_layers=None, prot_vec=None,
                 prot_len=1400, aa_dim=128, activation="relu", drug_len=2048, drug_vec="ECFP4",
                 protein_layers=None):

        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__chemberta_layers = (384, 128)
        self.__protein_strides = protein_windows
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__final_fc_layers = final_fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__aa_dim = aa_dim
        self.__drug_vec = drug_vec
        self.__drug_len = drug_len
        self.__drug_chemberta_len = 384
        self.__activation = activation
        self.__protein_layers = protein_layers
        self.__decay = decay
        self.__initializer = "glorot_normal"
        self.__regularizer_param = 0.001
        self.training_losses = {}
        self.training_losses["Training Loss"] = []
        self.training_losses["Validation Loss"] = []
        self.params_dic = {
            "kernel_initializer": self.__initializer,
            "kernel_regularizer": l2(self.__regularizer_param),
        }
        self.__model_t = self.modelv()
        opt = Adam(learning_rate=self.__learning_rate)       
        self.__model_t.compile(
            optimizer=opt,
            loss=variance_loss(lambda_var=0.1, delta=2.0),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
        )


    def update_model_weights(self, weights):
        self.__model_t.set_weights(weights)
    
    def return_tuple(self, value):
        if(isinstance(value, int)):
            return [value]
        else:
            return tuple(value)

    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            model_p = Conv1D(filters=filters, kernel_size=size, padding='same',
                                    kernel_initializer=initializer,
                                    kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def build_drug_branch(self, input_d):
        input_layer_d = input_d
        if(self.__drugs_layer is not None):
            for layer_size in self.return_tuple(self.__drugs_layer):
                model_d = Dense(layer_size, **self.params_dic)(input_layer_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(self.__activation)(model_d)
                model_d = Dropout(self.__dropout)(model_d)
                input_layer_d = model_d
        return input_layer_d

    
    def build_chemberta_branch(self, input_c):
        input_layer_c = input_c
        if(self.__chemberta_layers is not None):
            for layer_size in self.return_tuple(self.__chemberta_layers):
                model_c = Dense(layer_size, **self.params_dic)(input_layer_c)
                model_c = BatchNormalization()(model_c)
                model_c = Activation(self.__activation)(model_c)
                model_c = Dropout(self.__dropout)(model_c)
                input_layer_c = model_c
        return input_layer_c

    
        
    def build_protein_branch(self, input_p):
        if self.__prot_vec == "Convolution":
            # input_p shape: (batch_size, 1400, 128)
            model_p = Dense(20, **self.params_dic)(input_p)
            model_p = SpatialDropout1D(0.2)(model_p)

            model_ps = [
                self.PLayer(
                    stride,
                    self.__filters,
                    self.__activation,
                    self.__initializer,
                    self.__regularizer_param
                )(model_p)
                for stride in self.__protein_strides
            ]

            model_p = (
                Concatenate(axis=1)(model_ps)
                if len(model_ps) > 1
                else model_ps[0]
            )
        else:
            model_p = input_p

        if self.__protein_layers:
            input_layer_p = model_p

            for layer_size in self.return_tuple(self.__protein_layers):
                model_p = Dense(layer_size, **self.params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(self.__activation)(model_p)
                model_p = Dropout(self.__dropout)(model_p)
                input_layer_p = model_p

        return model_p


    
    def interaction_features(self, model_c, model_d, model_p):
        model_i = Concatenate(axis=1)([model_c, model_d, model_p])
        if(self.__fc_layers is not None):
            for fc_layer in self.return_tuple(self.__fc_layers):
                model_i = Dense(units=fc_layer, **self.params_dic)(model_i)
                model_i = BatchNormalization()(model_i)
                model_i = Activation(self.__activation)(model_i)
                model_i = Dropout(self.__dropout)(model_i)
        return model_i
                

    def modelv(self):
        input_chemberta = Input(shape=(self.__drug_chemberta_len,))
        input_d = Input(shape=(self.__drug_len,))
        input_p = Input(shape=(self.__prot_len,self.__aa_dim))
        model_c = self.build_chemberta_branch(input_chemberta)
        model_d = self.build_drug_branch(input_d)
        model_p = self.build_protein_branch(input_p)
        model_t = self.interaction_features(model_c, model_d, model_p)        
        if(self.__final_fc_layers is not None):
            for fc_layer in self.return_tuple(self.__final_fc_layers):
                model_t = Dense(units=fc_layer, **self.params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(self.__activation)(model_t)
                model_t = Dropout(self.__dropout)(model_t)       
        model_t = Dense(1, activation='linear', name='therm',
                          **self.params_dic)(model_t)
        model_f = Model(
            inputs=[input_chemberta, input_d, input_p],
            outputs=model_t
        )
        return model_f

    def feature_extractor(self):
        input_chemberta = Input(shape=(self.__drug_chemberta_len,))
        input_d = Input(shape=(self.__drug_len,))
        input_p = Input(shape=(self.__prot_len,self.__aa_dim))
        model_c = self.build_chemberta_branch(input_chemberta)
        model_d = self.build_drug_branch(input_d)
        model_p = self.build_protein_branch(input_p)
        model_t = self.interaction_features(model_c, model_d, model_p)
        model_f = Model(inputs=[input_chemberta, input_d, input_p], outputs=model_t)
        return model_f

    
    def fit(
        self,
        chemberta_features,
        drug_feature,
        protein_sequences,
        protein_embedding_dic,
        therm_values,
        sample_weight=None,
        batch_size=128
    ):
        n_samples = len(protein_sequences)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)

        def batch_generator():
            # Shuffle samples each epoch
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_sequences = protein_sequences[batch_idx]

                # Stack only the embeddings for this batch
                batch_protein = np.stack([
                    protein_embedding_dic[seq]
                    for seq in batch_sequences
                ]).astype(np.float32, copy=False)

                batch_c = np.asarray(
                    chemberta_features[batch_idx],
                    dtype=np.float32
                )
                batch_d = np.asarray(
                    drug_feature[batch_idx],
                    dtype=np.float32
                )
                batch_y = np.asarray(
                    therm_values[batch_idx],
                    dtype=np.float32
                ).reshape(-1)

                batch_w = np.asarray(
                    sample_weight[batch_idx],
                    dtype=np.float32
                )

                yield (
                    (batch_c, batch_d, batch_protein),
                    batch_y,
                    batch_w
                )

        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                (
                    tf.TensorSpec(
                        shape=(None, self.__drug_chemberta_len),
                        dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(None, self.__drug_len),
                        dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(None, self.__prot_len, self.__aa_dim),
                        dtype=tf.float32
                    )
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )

        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        history = self.__model_t.fit(
            dataset,
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            verbose=1
        )

        self.training_losses["Training Loss"].append(
            history.history["loss"][0]
        )

        return self.__model_t


    def summary(self):
        self.__model_t.summary()

    def save(self, output_file):
        self.__model_t.save(output_file)
        
    def load(self, input_file):
        self.__model_t = tf.keras.models.load_model(input_file, custom_objects={
            "loss": variance_loss(lambda_var=0.1, delta=2.0),
            "variance_loss": variance_loss})
    

    
    def predict(self, batch_size=128, **kwargs):
        results_dic = {}

        for dataset_name, data in kwargs.items():
            inference_result = self.inference(
                batch_size=batch_size,
                **{dataset_name: data},
            )

            results_dic[dataset_name] = {
                "therm_true": data["therm_values"],
                "predicted": inference_result[dataset_name]["predicted"],
            }

        return results_dic


    
    def inference(self, batch_size=128, **kwargs):
        results_dic = {}

        for dataset_name, data in kwargs.items():
            chemberta_features = data["chemberta_features"]
            drug_feature = data["drug_feature"]
            protein_sequences = data["protein_sequences"]
            protein_embedding_dic = data["protein_embedding_dic"]

            n_samples = len(protein_sequences)
            predictions = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                # Only construct protein embeddings for the current batch.
                protein_batch = np.stack([
                    protein_embedding_dic[seq]
                    for seq in protein_sequences[start:end]
                ]).astype(np.float32, copy=False)

                chemberta_batch = np.asarray(
                    chemberta_features[start:end], dtype=np.float32
                )

                drug_batch = np.asarray(
                    drug_feature[start:end], dtype=np.float32
                )

                # Pass all three inputs explicitly to Keras.
                batch_predictions = self.__model_t(
                    [
                        chemberta_batch,
                        drug_batch,
                        protein_batch,
                    ],
                    training=False,
                ).numpy()

                predictions.append(batch_predictions)

            results_dic[dataset_name] = {
                "predicted": np.concatenate(predictions, axis=0)
            }

        return results_dic

