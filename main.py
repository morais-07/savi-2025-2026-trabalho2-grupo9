#!/usr/bin/env python3
# shebang line for linux / mac

import glob
from html import parser
import os
from random import randint
import shutil
import signal
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse

import torch
# NOVO (Aponta para o teu ficheiro novo):
from dataset_new import Dataset

from torchvision import transforms
from model import ModelBetterCNN, ModelFullyconnected, ModelConvNet, ModelConvNet3
from trainer import Trainer
from datetime import datetime


def sigintHandler(signum, frame):
    print('SIGINT received. Exiting gracefully.')
    exit(0)


def main():

    # ------------------------------------
    # Setup argparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    # Argumento para a pasta do dataset
    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='./data',
                        help='Path where the dataset will be downloaded')

    # Argumento para a percentagem de exemplos a usar
    parser.add_argument('-pe', '--percentage_examples', type=float, default=1.0,
                        help='Percentage of examples to use for training and testing')

    # Argumento para o número de épocas
    parser.add_argument('-ne', '--num_epochs', type=int, default=10,
                        help='Number of epochs for training')

    # Argumento para o batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size for training and testing.')

    # Argumento para o experiment path
    parser.add_argument('-ep', '--experiment_path', type=str,
                        default='./experiments',
                        help='Path to save experiment results.')

    # Argumento para recomeçar o treino
    parser.add_argument('-rt', '--resume_training', action='store_true',
                        help='Resume training from last checkpoint if available.')

    # Argumento para escolher o modelo
    parser.add_argument('--model_type', type=str, default='BetterCNN', 
                        help='Escolha o modelo: FullyConnected, ConvNet, ConvNet3, BetterCNN')

    # Analisar os argumentos 
    args = vars(parser.parse_args())
    print(args)


    # ------------------------------------
    # Setup signal handler for exit on Ctrl+C
    # ------------------------------------
    signal.signal(signal.SIGINT, sigintHandler) # O sigintHandler é o que faz o exit(0), ou seja, sai do programa quando se carrega em Ctrl+C


    # ------------------------------------
    # Create the experiment
    # ------------------------------------

    # VERIFICAR NO FINAL SE VALE A PENA CRIAR ESTAS PASTAS COM O NOME DA DATA
    # experiment_name = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # experiment_name = datetime.today().strftime('%Y-%m-%d %H')  # same experiment every hour
    # experiment_name = datetime.today().strftime('%Y-%m-%d %H')  # same experiment every hour
    # args['experiment_full_name'] = os.path.join(
    #     args['experiment_path'], experiment_name)
    args['experiment_full_name'] = args['experiment_path']

    print('Starting experiment: ' + args['experiment_full_name'])

    # VERIFICAR SE É MESMO NECESSÁRIO ISTO COMENTADO - FOI COMENTADO PARA NÃO APAGAR OS RESULTADOS, MAS PODE SER ÚTIL PARA OS APAGAR AUTOMATICAMENTE
    # if os.path.exists(args['experiment_full_name']):
    #     shutil.rmtree(args['experiment_full_name'])
    #     print('Experiment folder already exists. Deleting to start fresh.')

    os.makedirs(args['experiment_full_name'], exist_ok=True)


    # ------------------------------------
    # Create datasets
    # ------------------------------------

    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)


    # ------------------------------------
    # Create the model
    # ------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define o dispositivo (GPU se disponível, senão CPU)
    print(f'Using device: {device}')

    # Verifica qual o modelo escolhido nos argumentos
    if args['model_type'] == 'FullyConnected':
        model = ModelFullyconnected()
    elif args['model_type'] == 'ConvNet':
        model = ModelConvNet()
    elif args['model_type'] == 'ConvNet3':
        model = ModelConvNet3()
    elif args['model_type'] == 'BetterCNN':
        model = ModelBetterCNN()
    else:
        # Se não for nenhum dos modelos, dá erro
        raise ValueError(f"Modelo desconhecido: {args['model_type']}")

    model = model.to(device) # Enviar o modelo escolhido para o device (GPU/CPU)
    args['device'] = device # Adicionar o device aos argumentos para usar no Trainer


    # ------------------------------------
    # Start training
    # ------------------------------------

    trainer = Trainer(args, train_dataset, test_dataset, model) # Cria o Trainer
    trainer.train() # Treina o modelo
    trainer.evaluate() # Avalia o modelo no dataset de teste


if __name__ == '__main__':
    main()
