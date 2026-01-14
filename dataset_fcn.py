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
from torch.utils.data import Dataset
import cv2

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

# --- CLASSE PARA A VERSÃO A (1 dígito por linha) ---
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
                        self.data.append(parts) # Guarda a linha completa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = self.data[idx]
        img_path = os.path.join(self.images_dir, parts[0])

        # --- SOLUÇÃO PARA ACENTOS NO WINDOWS ---
        # Em vez de cv2.imread, usamos numpy para ler o ficheiro e depois descodificamos
        try:
            # np.fromfile lê os bytes brutos do disco
            img_array = np.fromfile(img_path, dtype=np.uint8)
            # cv2.imdecode converte esses bytes numa imagem
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            image = None

        # Verificação de segurança caso a imagem continue a não ser lida
        if image is None:
            print(f"Erro crítico ao ler: {img_path}")
            return torch.zeros((1, 128, 128)), torch.full(self.output_size, 10).long()
        # ---------------------------------------

        h, w = image.shape # Deve ser 128x128
        target_map = np.full((h, w), 10, dtype=np.int64)

        # Lógica para Versão A (1 dígito) ou adaptar para Versão D
        # Se estiveres na Classe da Versão A:
        cls, x, y, dw, dh = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        target_map[y:y+dh, x:x+dw] = cls

        # Redimensionar e converter para tensor
        target_res = cv2.resize(target_map, self.output_size, interpolation=cv2.INTER_NEAREST)
        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target_res).long()

        unique_classes = np.unique(target_res)
        print(f"Classes no target: {unique_classes}")

        return img_tensor, target_tensor

# --- CLASSE PARA A VERSÃO D (Vários dígitos por linha) ---
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
        
        # --- SOLUÇÃO PARA ACENTOS (imdecode) ---
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            image = None

        # Verificação de segurança
        if image is None:
            print(f"Erro ao ler imagem na Versão D: {img_path}")
            # Retorna tensores "dummy" para não quebrar o batch do treino
            return torch.zeros((1, 128, 128)), torch.full(self.output_size, 10).long()

        # O mapa de alvo deve ter o mesmo tamanho da imagem original antes do resize
        target_map = np.full((128, 128), 10, dtype=np.int64)
        
        # Loop para processar todos os dígitos da linha
        # i:i+5 extrai (classe, x, y, largura, altura)
        for i in range(2, len(parts), 5):
            try:
                # Usamos map(float, ...) e depois int(...) para evitar erros se houver decimais
                cls, x, y, dw, dh = map(lambda x: int(float(x)), parts[i:i+5])
                
                # Garantir que as coordenadas não saem fora dos limites da imagem (0-127)
                y1, y2 = max(0, y), min(128, y + dh)
                x1, x2 = max(0, x), min(128, x + dw)
                
                target_map[y1:y2, x1:x2] = cls
            except:
                continue # Se houver um erro numa caixa específica, salta para a próxima

        # Redimensionar o mapa de classes (manter INTER_NEAREST para não criar classes inexistentes)
        target_res = cv2.resize(target_map, self.output_size, interpolation=cv2.INTER_NEAREST)
        
        # Converter para Tensores
        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target_res).long()

        return img_tensor, target_tensor