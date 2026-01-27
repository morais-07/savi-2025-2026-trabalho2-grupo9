import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset

#Versão A - 1 dígito por imagem 
class SceneDatasetA(Dataset):
    def __init__(self, root_dir, subset='train', output_size=(32, 32)):
        self.images_dir = os.path.join(root_dir, subset, 'images')
        self.labels_file = os.path.join(root_dir, subset, 'labels.txt')
        self.output_size = output_size
        self.data = []
        
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 7:
                        self.data.append(parts) #guarda a linha completa

    def __len__(self):
        return len(self.data) #nº de imagens no treino

    def __getitem__(self, idx):  #idx é um valor de 0 até len-1
        parts = self.data[idx]
        img_path = os.path.join(self.images_dir, parts[0])    #associamos o label da imagem ao path

        #para evitar problemas com acentos
        try:
            #np.fromfile lê os bytes brutos do disco
            img_array = np.fromfile(img_path, dtype=np.uint8)
            #cv2.imdecode converte esses bytes numa imagem
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            image = None

        #caso a imagem continue a não ser lida, envia uma imagem vazia para o treino continuar
        if image is None:
            print(f"Erro crítico ao ler: {img_path}")
            return torch.zeros((1, 128, 128)), torch.full(self.output_size, 10).long()

        h, w = image.shape # Deve ser 128x128
        target_map = np.full((h, w), 10, dtype=np.int64) #criação de uma matriz com o mesmo tamanho da imagem, preenchida com 10 (fundo), nºs inteiros

        #Versão A 
        cls, x, y, dw, dh = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        target_map[y:y+dh, x:x+dw] = cls  #preenche na posição do dígito a classe correspondente

        #Redimensionar para tamanho de output e converter para tensor (imagem de 128x128, mas a saída da FCN é geralmente 32x32)
        target_res = cv2.resize(target_map, self.output_size, interpolation=cv2.INTER_NEAREST)  #INTER_NEAREST mantém classes inteiras
        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0  #nºs decimais, adiciona canal de cor, converte para 0-1
        target_tensor = torch.from_numpy(target_res).long()

        unique_classes = np.unique(target_res)  #'resume' o conteúdo 
        print(f"Classes no target: {unique_classes}") #para a versão A deve aparecer o dígito e o fundo

        return img_tensor, target_tensor

#Versão D - 3 a 5 dígitos de diferentes tamanhos
class SceneDatasetD(Dataset):
    def __init__(self, root_dir, subset='train', output_size=(32, 32)):
        self.images_dir = os.path.join(root_dir, subset, 'images')
        self.labels_file = os.path.join(root_dir, subset, 'labels.txt')
        self.output_size = output_size
        self.data = []

        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 2:
                        self.data.append(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = self.data[idx]
        img_path = os.path.join(self.images_dir, parts[0])
        
        #para evitar problemas com acentos
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            image = None

        
        if image is None:
            print(f"Erro ao ler imagem na Versão D: {img_path}")
            return torch.zeros((1, 128, 128)), torch.full(self.output_size, 10).long()

        #criação do target map
        target_map = np.full((128, 128), 10, dtype=np.int64)
        
        #Versão D
        for i in range(2, len(parts), 5): #começamos no idx 2 (info sobre classe) e avançamos de 5 em 5 até ao final
            try:
                cls, x, y, dw, dh = map(lambda x: int(float(x)), parts[i:i+5]) #map aplica a regra a todos os valores
                
                #garantir que as coordenadas não saem dos limites da imagem (0-127)
                y1, y2 = max(0, y), min(128, y + dh)
                x1, x2 = max(0, x), min(128, x + dw)
                
                target_map[y1:y2, x1:x2] = cls #colocar na área específica a classe do dígito
            except:
                continue 

        #Redimensionar 
        target_res = cv2.resize(target_map, self.output_size, interpolation=cv2.INTER_NEAREST)
        
        #Converter para tensores
        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target_res).long()

        return img_tensor, target_tensor