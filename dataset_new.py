import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms, datasets # New datasets import

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):

        # Store the arguments in class properties
        self.args = args
        self.train = is_train

        # ---------------------------------
        # NOVO: Configuração para torchvision (Tarefa 1)
        # ---------------------------------
        # Definimos as transformações necessárias (Converter para Tensor + Normalizar)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalização padrão do MNIST (ajuda a rede a aprender mais rápido)
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])

        # Download e carregamento automático
        # Isto substitui toda a parte de ler ficheiros e labels.txt manualmente
        self.data_source = datasets.MNIST(
            root=args['dataset_folder'], 
            train=is_train, 
            download=True, 
            transform=self.transform
        )

        # ---------------------------------
        # Lógica de Percentagem (Adaptada)
        # ---------------------------------
        # Em vez de cortar listas, definimos o tamanho virtual do dataset
        self.num_examples = int(len(self.data_source) * args['percentage_examples'])
        
        # O antigo:
        # self.image_filenames = self.image_filenames[0:num_examples]
        # self.labels = self.labels[0:num_examples]

    def __len__(self):
        # This function returns the number of examples in the dataset
        return self.num_examples

    def __getitem__(self, idx):
        # ----------------------------
        # NOVO: Obter dados do torchvision
        # ----------------------------
        # O dataset do torch devolve (imagem, label_inteiro)
        # A imagem já vem transformada em Tensor por causa do self.transform lá em cima
        image_tensor, label_index = self.data_source[idx]

        # ----------------------------
        # Manter compatibilidade com MSELoss (One-Hot Encoding)
        # ----------------------------
        # Aqui mantivemos a lógica exata para transformar um número num vetor
        
        label = [0]*10  # cria uma lista de zeros com tamanho 10
        label[label_index] = 1  # define o índice correspondente ao dígito como 1

        label_tensor = torch.tensor(label, dtype=torch.float)

        # ----------------------------
        # ANTIGO (Comentado)
        # ----------------------------
        # label_index = int(self.labels[idx])
        # label = [0]*10 
        # label[label_index] = 1 
        # label_tensor = torch.tensor(label, dtype=torch.float)

        # image_filename = self.image_filenames[idx]
        # image = Image.open(image_filename).convert('L') 
        # image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor