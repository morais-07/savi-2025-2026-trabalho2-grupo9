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

#Carregar dados do mnist
args = {'dataset_folder': './data', 'percentage_examples': 1.0} #carregar 100% dos dados
#MUDADO TINHAS UM ARGUMENTO DO TEU PC

# Define quantas imagens queres gerar
NUM_TRAIN = 5000
NUM_TEST = 1000

print("A carregar dígitos originais...")

to_pil = transforms.ToPILImage() #converte tensor em imagem 

mnist_train = Dataset(args, is_train=True)  #objeto com listas 
mnist_test = Dataset(args, is_train=False)

print(f"Dígitos de treino disponíveis: {len(mnist_train)}")
print(f"Dígitos de teste disponíveis: {len(mnist_test)}")

#canvas = Image.new('L',(128,128),0) #criar imagem preta (0) com 128x128´(pedido no enunciado)
#criar dataset novo Versão A - 1 dígito aleatório por imagem, dataset com pasta train e test
ds_A = 'Dataset_Cenas_Versão_A'
ds_D = 'Dataset_Cenas_Versão_D'

os.makedirs(os.path.join(ds_A, 'train', 'images'), exist_ok=True)
labels_path1A = os.path.join(ds_A, 'train', 'labels.txt')
f_labels1A = open(labels_path1A, 'w')

os.makedirs(os.path.join(ds_A, 'test', 'images'), exist_ok=True)
labels_path2A = os.path.join(ds_A, 'test', 'labels.txt')
f_labels2A = open(labels_path2A, 'w')


os.makedirs(os.path.join(ds_D, 'train', 'images'), exist_ok=True)
labels_path1D = os.path.join(ds_D, 'train', 'labels.txt')
f_labels1D = open(labels_path1D, 'w')

os.makedirs(os.path.join(ds_D, 'test', 'images'), exist_ok=True)
labels_path2D = os.path.join(ds_D, 'test', 'labels.txt')
f_labels2D = open(labels_path2D, 'w')

#Ao criar uma nova imagem, adicionar ao ficheiro de labels: nome da imagem, numero de digitos, label, posição (x,y) e size (w,h)

#criar dataset novo Versão D - múltiplos dígitos por imagem (sem diferença de escala), dataset com pasta train e test
#VERSÃO A
#Criar ciclo para 5000 imagens de treino
#para cada ciclo: criar imagem preta 
#tirar imagem aleatória da pool de 60000
#reduzir para tamanho entre 22x22 e 36x36
#colar em posição aleatória da imagem preta
#guardar num dataset novo versão a train
#Criar ciclo para 1000 imagens de teste 
#labels_path = os.path.join(ds_A, 'train', 'labels.txt') #criar ficheiro para labels
for i in range(NUM_TRAIN):
    canvas = Image.new('L',(128,128),0) #criar imagem 
    #escolher imagem aleatória 
    idx = random.randint(0,len(mnist_train)-1) #escolher um indice random desde 0 a 4999. len(mnist_tran)=5000, e o range vai de 0 a 4999

    img_tensor, label_tensor = mnist_train[idx]
    label = torch.argmax(label_tensor).item()
    img_digito = to_pil(img_tensor)

    #img_caminho = mnist_train.image_filenames[idx] #retirar path do indice aleatório 
    #label = mnist_train.labels[idx] #retirar label da imagem de indice aleatório

    #Abrir e redimensionar o dígito para uma escala entre 22 e 26
    #img_digito = Image.open(img_caminho) #como retirámos o path temos de abrir a imagem
    s = random.randint(22,36) #num random entre 22 e 36 para a escala
    img_digito=img_digito.resize((s,s))
    img_digito = img_digito.point(lambda p: p if p > 155 else 0) # Filtra o "lixo" cinzento
    #FALTA FAZER A CORREÇÃO DA COR PQ OBVIAMENTE N ESTÁ BEM
    #calcular posição aleatória na imagem preta 
    w, h = img_digito.size #valor de largura e altura do digito
    x = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
    y = random.randint(0,128-h)
    #Colar a imagem na posição 
    #Guardar imagem 
    canvas.paste(img_digito,(x,y), img_digito)
    nome_ficheiro = f"img_{i}.jpg"
    new_path = os.path.join(ds_A,'train','images',nome_ficheiro)
    canvas.save(new_path)
    #Guardar Label para estatísticas
    linha = f"{nome_ficheiro} 1 {int(label)} {x} {y} {w} {h}\n"
    f_labels1A.write(linha)

f_labels1A.close()
for i in range(NUM_TEST):
    canvas = Image.new('L',(128,128),0)

    idx = random.randint(0,len(mnist_test)-1)

    img_tensor, label_tensor = mnist_test[idx]
    label = torch.argmax(label_tensor).item()
    img_digito = to_pil(img_tensor)
    
    s = random.randint(22,36) #num random entre 22 e 36 para a escala
    img_digito=img_digito.resize((s,s))
    img_digito = img_digito.point(lambda p: p if p > 155 else 0) # Filtra o "lixo" cinzento
    #FALTA FAZER A CORREÇÃO DA COR PQ OBVIAMENTE N ESTÁ BEM
    #calcular posição aleatória na imagem preta 
    w, h = img_digito.size #valor de largura e altura do digito
    x = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
    y = random.randint(0,128-h)
    #Colar a imagem na posição 
    canvas.paste(img_digito,(x,y), img_digito)
    nome_ficheiro = f"img_{i}.jpg"
    new_path = os.path.join(ds_A,'test','images',nome_ficheiro)
    canvas.save(new_path)
    #Guardar Label para estatísticas
    linha = f"{nome_ficheiro} 1 {int(label)} {x} {y} {w} {h}\n"
    f_labels2A.write(linha)
    
f_labels2A.close()
print("Criação de Cenas - Versão A concluida")

#VERSÃO D
#Criar imagens com multiplos dígitos (3 a 5) com tamanhos variados e sem sobreposição
#TRAIN
for i in range(NUM_TRAIN):
    canvas = Image.new('L',(128,128),0)
    n = random.randint(3,5) #escolha random do nº de digitos da imagem~
    idx=[]  #criação de lista para os índices
    labels1D = [] #criação da lista para labels desta imagem
    boxes = [] #lista de posições ocupadas
    for e in range(n): 
        indice = random.randint(0,len(mnist_train)-1) #escolher um indice random
        idx.append(indice)      #adicionar à lista de indices
    #quando este ciclo for termina, temos uma lista idx com os indices random escolhidos
    #temos que ler as imagens destes indices, dar resize para random (22,36), colar em posição aleatória na imagem preta
    #guardar num dataset da versão C train
    num = len(idx)
    for u in range(num):
        ind = idx[u]
        img_tensor , label_tensor = mnist_train[ind]
        label = torch.argmax(label_tensor).item()
        img_digito = to_pil(img_tensor)

        s = random.randint(22,36) #num random entre 22 e 36 para a escala
        img_digito=img_digito.resize((s,s))
        img_digito = img_digito.point(lambda p: p if p > 155 else 0) # Filtra o "lixo" cinzento
        #FALTA FAZER A CORREÇÃO DA COR PQ OBVIAMENTE N ESTÁ BEM
        #calcular posição aleatória na imagem preta 
        w, h = img_digito.size #valor de largura e altura do digito

        #x = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
        #y = random.randint(0,128-h)
        #Posição candidata

        colidiu = True
        tentativas = 0

        while colidiu and tentativas<100:
            x_cand = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
            y_cand = random.randint(0,128-h)
            #definir bordas limite do novo digito
            x_max_cand = x_cand + w
            y_max_cand = y_cand + h #MUDADO, TINHAS Y EM VEZ DE H AQUI

            sobrepõe = False
            for (bx_min, by_min, bx_max, by_max) in boxes:
                # Lógica de colisão: se os retângulos se cruzarem em ambos os eixos
                if not (x_max_cand <= bx_min or x_cand >= bx_max or y_max_cand <= by_min or y_cand >= by_max):
                    sobrepõe = True
                    break
            
            if not sobrepõe:
                # Se não bate em nada, guarda as coordenadas e sai do while
                x, y = x_cand, y_cand
                boxes.append((x, y, x + w, y + h))
                colidiu = False
            
            tentativas += 1

        # Só cola se tiver encontrado um lugar sem colisão
        if not colidiu:
            canvas.paste(img_digito, (x, y), img_digito)
            labels1D.append(f"{label} {x} {y} {w} {h}")


        #Colar a imagem na posição 
        #canvas.paste(img_digito,(x,y), img_digito)
        


    nome_ficheiro = f"img_{i}.jpg"
    new_path = os.path.join(ds_D,'train','images',nome_ficheiro)
    canvas.save(new_path)
    
    info_total = " ".join(labels1D)
    f_labels1D.write(f"{nome_ficheiro} {len(labels1D)} {info_total}\n")

f_labels1D.close()

 #TEST
for i in range(NUM_TEST):
    canvas = Image.new('L',(128,128),0)
    n = random.randint(3,5) #escolha random do nº de digitos da imagem~
    idx=[]  #criação de lista para os índices
    labels2D = []
    boxes = [] #lista de posições ocupadas
    for e in range(n): 
        indice = random.randint(0,len(mnist_test)-1) #escolher um indice random
        idx.append(indice)      #adicionar à lista de indices
    #quando este ciclo for termina, temos uma lista idx com os indices random escolhidos
    #temos que ler as imagens destes indices, dar resize para random (22,36), colar em posição aleatória na imagem preta
    #guardar num dataset da versão C train
    num = len(idx)
    for u in range(num):
        ind = idx[u]
        img_tensor , label_tensor = mnist_test[ind]
        label = torch.argmax(label_tensor).item()
        img_digito = to_pil(img_tensor)

        s = random.randint(22,36) #num random entre 22 e 36 para a escala
        img_digito=img_digito.resize((s,s))
        img_digito = img_digito.point(lambda p: p if p > 155 else 0) # Filtra o "lixo" cinzento
        #FALTA FAZER A CORREÇÃO DA COR PQ OBVIAMENTE N ESTÁ BEM
        #calcular posição aleatória na imagem preta 
        w, h = img_digito.size #valor de largura e altura do digito

        colidiu = True
        tentativas = 0

        while colidiu and tentativas<100:
            x_cand = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
            y_cand = random.randint(0,128-h)
            #definir bordas limite do novo digito
            x_max_cand = x_cand + w
            y_max_cand = y_cand + y

            sobrepõe = False
            for (bx_min, by_min, bx_max, by_max) in boxes:
                # Lógica de colisão: se os retângulos se cruzarem em ambos os eixos
                if not (x_max_cand <= bx_min or x_cand >= bx_max or y_max_cand <= by_min or y_cand >= by_max):
                    sobrepõe = True
                    break
            
            if not sobrepõe:
                # Se não bate em nada, guarda as coordenadas e sai do while
                x, y = x_cand, y_cand
                boxes.append((x, y, x + w, y + h))
                colidiu = False
            
            tentativas += 1

        # Só cola se tiver encontrado um lugar sem colisão
        if not colidiu:
            canvas.paste(img_digito, (x, y), img_digito)
            labels2D.append(f"{label} {x} {y} {w} {h}")

        #Colar a imagem na posição 
        #canvas.paste(img_digito,(x,y), img_digito)
        #x = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
        #y = random.randint(0,128-h)
        

    nome_ficheiro = f"img_{i}.jpg"
    new_path = os.path.join(ds_D,'test','images',nome_ficheiro)
    canvas.save(new_path)
    #Guardar Label para estatísticas
    info_total = " ".join(labels2D)
    f_labels2D.write(f"{nome_ficheiro} {len(labels2D)} {info_total}\n")

f_labels2D.close()
print("Criação de Cenas - Versão D concluida")

