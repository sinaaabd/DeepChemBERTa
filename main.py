'''
DeepChemBERTa: Compound-based Protein Stabilization Prediction
Author: Dr. Sina Abdollahi
'''
from deepchemberta_utils import *
from DeepChemBERTa import *
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import time
import sys
import os




if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset', default='Test')
    parser.add_argument("--load_model", "-l", help="load an existing model", type=str)
    parser.add_argument("--save_model", "-s", help="Save the model's name", type=str)
    parser.add_argument("--train", "-T", help="express training CSV file", type=str)
    parser.add_argument("--test", "-t", help="express test CSV file", type=str)
    parser.add_argument("--val", "-v", help="express validation CSV file", type=str)
    parser.add_argument("--inference", "-i", help="express inference CSV file", type=str)

    args = parser.parse_args()
    data_path = f"datasets/{args.data}"
    result_path = f"results/{args.data}"

    if os.path.exists(result_path):
        print(f"Result file for {result_path} exists, skipping...")
    else:
        print(f"Result file for {result_path} does not exist, deleting...")
        os.makedirs(result_path, exist_ok=True)
    result_df = run_deepchemberta(data_path, result_path, args.load_model, args.save_model, args.train, args.val, args.test, args.inference)
    end_time = time.time()
    print(f"\tThe whole program EXECUTION TIME: {end_time - start_time}")
