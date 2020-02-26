import os
import shutil
import glob

import torch
import torch.nn as nn
import argparse
import numpy as np

from models.HiCSR_model import Generator as HiCSR
from models.DAE_model import DAE

from dataloader import DatasetFromFolder
from torch.utils.data import DataLoader

def predict(model, data, input_size=40, output_size=28):
    left_pad_value = int((input_size - output_size)/2)
    right_pad_value = left_pad_value + (output_size - (data.shape[0]%output_size))
    padded_data = np.pad(data, ((left_pad_value,right_pad_value),(left_pad_value,right_pad_value)), mode='constant')
    predicted_mat = np.zeros((padded_data.shape[0], padded_data.shape[1]))
    for i in range(0,data.shape[0], output_size):
        for j in range(0,data.shape[1], output_size):
            block_numpy = padded_data[i:i+input_size,j:j+input_size]
            block_torch = torch.from_numpy(block_numpy).view(1,1,input_size,input_size).float().to(device)
            predicted_mat[i:i+output_size,j:j+output_size] = model(block_torch).detach().cpu().numpy()[0][0]

    # zero out rows and cols with no signal
    zero_cols = np.where(np.sum(data, axis=0) == -1*data.shape[0])
    zero_rows = np.where(np.sum(data, axis=1) == -1*data.shape[0])
    predicted_mat[:,zero_cols] = -1
    predicted_mat[zero_rows,:] = -1

    predicted_mat = predicted_mat[0:data.shape[0], 0:data.shape[1]]

    # force symmetry if square
    if predicted_mat.shape[0] == predicted_mat.shape[1]:
        predicted_mat = np.triu(predicted_mat) + np.triu(predicted_mat, 1).T
        assert np.allclose(predicted_mat, predicted_mat.T, rtol=1e-5, atol=1e-8)

    return predicted_mat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="input directory where the low resolution Hi-C matrices to be enhanced are stored")
    parser.add_argument('--output', type=str, required=True, help="output directory to store resultant Hi-C matrices are stored")
    parser.add_argument('--model_type', type=str, required=True, help="model type to predict with, HiCSR or DAE (Denoising Autoencoder). HiCSR is used for Hi-C enhancement, DAE is the loss network used to train HiCSR")
    parser.add_argument('--model_fp', type=str, required=True, help="pytorch model filepath to load for enhancement predictions")
    args = parser.parse_args()

    if os.path.exists(args.output):
        print("output path already exists. Overwrite? (y/n)")
        overwrite = input()
        if overwrite.lower() == 'y':
            shutil.rmtree(args.output)
            os.mkdir(args.output)
            os.mkdir(args.output+'{}'.format(args.model_type))
        else:
            exit()

    else:
        os.mkdir(args.output)
        os.mkdir(args.output+'{}'.format(args.model_type))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Predicting with GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print("Predicting with CPU")

    pred_paths =  glob.glob(args.input+'/*')

    if args.model_type == 'HiCSR':
        HiCSR = HiCSR(num_res_blocks=15)
        HiCSR.load_state_dict(torch.load(args.model_fp, map_location=device))
        HiCSR.to(device).eval()
        for pred_path in pred_paths:
            print("predicting {} with {}".format(pred_path, args.model_type))
            mat_lr = np.loadtxt(pred_path)
            HiCSR_predicted = predict(HiCSR, data=mat_lr, input_size=40, output_size=28)
            chromosome = pred_path.split('/')[-1].split('-')[0]
            cell_type = pred_path.split('/')[-1].split('-')[1]
            np.savetxt(args.output+'{}-{}-16-HiCSR.txt.gz'.format(chromosome, cell_type), HiCSR_predicted, delimiter=' ', fmt='%.8f')

    elif args.model_type == 'DAE':
        dae = DAE()
        dae.load_state_dict(torch.load(args.model_fp, map_location=device))
        dae.to(device).eval()
        for pred_path in pred_paths:
            print("predicting {} with {}".format(pred_path, args.model_type))
            mat_hr = np.loadtxt(pred_path)
            dae_predicted = predict(dae, data=mat_hr, input_size=40, output_size=40)
            chromosome = pred_path.split('/')[-1].split('-')[0]
            cell_type = pred_path.split('/')[-1].split('-')[1]
            np.savetxt(args.output+'{}-{}-1-DAE.txt.gz'.format(chromosome, cell_type), dae_predicted, delimiter=' ', fmt='%.8f')

    else:
        raise ValueError

