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
import argparse
from dataset_new import Dataset
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

#Tarefa 2 - Gerar visualizações e estatísticas

def analyze_dataset(dataset_path):
    images_dir = os.path.join(dataset_path, 'train', 'images')
    labels_file = os.path.join(dataset_path, 'train', 'labels.txt')
    
    # Listas para guardar estatísticas
    all_labels = []
    digits_per_image = []
    all_sizes = []
    
    # Lista para visualização de imagens em mosaico
    visualization_data = []

    with open(labels_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6: continue  #se for uma linha vazia ignorar e passar à próxima
            
            img_name = parts[0]
            
            num_digits = int(parts[1]) 
            cursor = 2                  #colocar o cursor onde começam os dados do dígito (pos 2)
            
            #adicona nº de digitos
            digits_per_image.append(num_digits) 
            img_bboxes = []
            
            #para cada dígito retirar a info sobre o que digito é e as coordenadas
            for _ in range(num_digits): 
                try:
                    label = int(parts[cursor])  
                    x = int(parts[cursor+1])
                    y = int(parts[cursor+2])
                    w = int(parts[cursor+3])
                    h = int(parts[cursor+4])
                
                    all_labels.append(label) #adicona qual é o digito 
                    all_sizes.append((w, h)) #adicona tamanho (w,h)
                    img_bboxes.append({'label': label, 'bbox': (x, y, w, h)}) #lista com label, coordenadas, posição
                    
                    cursor += 5 # salta para o próximo bloco de 5 valores
                except IndexError:
                    print(f"Erro na linha da imagem {img_name}. Formato inesperado.")
                    break
            
            # guardamos as primeiras 9 imagens para visualizar
            if len(visualization_data) < 9:
                visualization_data.append((img_name, img_bboxes))

    #Visualizar mosaico de 9 imagens
    #Definições de Visualização
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Visualização: {dataset_path}', fontsize=16)
    
    for i, (img_name, bboxes) in enumerate(visualization_data):
        ax = axs[i//3, i%3] #converter i em coordenadas (linha,coluna), // nº inteiro, % resto da divisão
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        
        for item in bboxes:
            label = item['label']
            x, y, w, h = item['bbox']
            #desenha o retângulo (x, y, largura, altura)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-2, str(label), color='red', fontsize=10, fontweight='bold')
        
        ax.set_title(img_name)
        ax.axis('off')
    
    plt.tight_layout() #ajusta espaçamento para que os títulos não se sobreponham 
    plt.show()

    #Estatísticas
    fig_stats, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    fig_stats.suptitle(f"Estatísticas: {dataset_path}", fontsize=14, fontweight='bold')
    
    #Distribuição de Classes 
    class_counts = Counter(all_labels)
    class_counts = Counter(all_labels)
    ax1.bar(class_counts.keys(), class_counts.values())
    ax1.set_title("Distribuição de Classes (0-9)")
    ax1.set_xticks(range(10))

    #Histograma de dígitos por imagem
    ax2.hist(digits_per_image, bins=range(min(digits_per_image), max(digits_per_image) + 2), align='left', rwidth=0.8)
    ax2.set_title("Número de dígitos por imagem")
    ax2.set_xticks(sorted(list(set(digits_per_image))))

    #Tamanho médio dos dígitos
    avg_w = np.mean([s[0] for s in all_sizes])
    avg_h = np.mean([s[1] for s in all_sizes])
    ax3.axis('off') #Esconde os eixos para mostrar só o texto
    texto = f"Tamanho médio dos dígitos: {avg_w:.2f} x {avg_h:.2f} pixels"
    ax3.text(0.5, 0.5, texto, fontsize=15, ha='center', fontweight='bold', bbox=dict(facecolor='none', edgecolor='black', pad=10))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


analyze_dataset('Dataset_Cenas_Versão_A')
analyze_dataset('Dataset_Cenas_Versão_D')