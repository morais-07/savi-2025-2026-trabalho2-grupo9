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


#Tarefa 2 - Criar novo dataset
#(Versão A - 1 dígito aleatório por imagem, Versão D - De 3 a 5 dígitos aleatórios por imagem)

#Carregar dados do mnist
args = {'dataset_folder': './data', 'percentage_examples': 1.0} #carregar 100% dos dados

# Define quantas imagens queres gerar
NUM_TRAIN = 5000
NUM_TEST = 1000

print("A carregar dígitos originais...")

to_pil = transforms.ToPILImage() #converte tensor em imagem, ACHO QUE POSSO APAGAR VERIFICAR

#atribuição de variáveis a cada dataset 
mnist_train = Dataset(args, is_train=True)  
mnist_test = Dataset(args, is_train=False)

#Quantidade de imagens disponíveis para treino e teste
print(f"Dígitos de treino disponíveis: {len(mnist_train)}")
print(f"Dígitos de teste disponíveis: {len(mnist_test)}")

#Nomeação de pastas onde se irão guardar as imagens criadas
ds_A = 'Dataset_Cenas_Versão_A'
ds_D = 'Dataset_Cenas_Versão_D'

#Cria pasta Versão A / treino / labels e imagens
os.makedirs(os.path.join(ds_A, 'train', 'images'), exist_ok=True)
labels_path1A = os.path.join(ds_A, 'train', 'labels.txt')
f_labels1A = open(labels_path1A, 'w')  #abre o documento txt para podermos escrever (w - write)

#Cria pasta Versão A / teste / labels e imagens
os.makedirs(os.path.join(ds_A, 'test', 'images'), exist_ok=True)
labels_path2A = os.path.join(ds_A, 'test', 'labels.txt')
f_labels2A = open(labels_path2A, 'w')

#Cria pasta Versão D / treino / labels e imagens
os.makedirs(os.path.join(ds_D, 'train', 'images'), exist_ok=True)
labels_path1D = os.path.join(ds_D, 'train', 'labels.txt')
f_labels1D = open(labels_path1D, 'w')

#Cria pasta Versão D / teste / labels e imagens
os.makedirs(os.path.join(ds_D, 'test', 'images'), exist_ok=True)
labels_path2D = os.path.join(ds_D, 'test', 'labels.txt')
f_labels2D = open(labels_path2D, 'w')

#VERSÃO A
#TRAIN
for i in range(NUM_TRAIN):     #o numero de imagens a gerar diz-nos o nº de vezes que vamos repetir este ciclo
    canvas = Image.new('L',(128,128),0) #criar imagem completamente preta (0) de 128x128 pixels
    idx = random.randint(0,len(mnist_train)-1) #escolher um indice random desde 0 a 4999. len(mnist_tran)=5000, e o range vai de 0 a 4999

    img_tensor, label_tensor = mnist_train[idx] #tirar o label e a imagem correspondente ao indice aleatório
    label = torch.argmax(label_tensor).item() #torch.argmax para tirar de one-hot encoding, .item() extrai o nº e trasnforma num num int 
    

    #Retirar os valores da imagem, guardar na cpu, converter para numpy, squeeze() elimina o nº de canais de cor 
    img_np = img_tensor.detach().cpu().numpy().squeeze()

    #Normalizar para garantir que o fundo seja 0 e o dígito vá até 255
    img_min = img_np.min()
    img_max = img_np.max()
    img_np = (img_np - img_min) / (img_max - img_min + 1e-8) # escala 0 a 1
    #img_np - img_min garante que o pixel mais escuro passe a ser exatamente 0
    #img_max-img_min é o intervalo de brilho, comprimimos os nºs para que fiquem entre 0 e 1.

    #Aplicamos um 'filtro', tudo o que tiver brilho menor que 20%, fica preto (0)
    img_np[img_np<0.2]=0
    #Agora convertemos para imagem PIL, a imagem está entre 0 e 1, multiplicamos por 255 para ficar nessa escala
    img_digito = Image.fromarray((img_np*255).astype(np.uint8), mode='L')


    #Redimensionar o dígito para uma escala entre 22 e 26
    #Se fosse Versão B:
    #s = random.randint(22,36) #num random entre 22 e 36 para a escala
    #Versão A (apenas 1 escala e posição aleatória)
    s = 28
    img_digito=img_digito.resize((s,s), Image.Resampling.LANCZOS) 
    #Image.Resampling.LANCZOS é um filtro de resampling (define como o computador inventa/funde píxeis)

    #calcular posição aleatória na imagem preta 
    w, h = img_digito.size #valor de largura e altura do digito
    x = random.randint(0,128-w) #a posição do digito no canva tem de ser de 0 a (128-w) porque um valor maior que este colocaria o digito fora da imagem preta
    y = random.randint(0,128-h)

    #Colar a imagem na posição x,y
    canvas.paste(img_digito,(x,y), img_digito) #img_digito é a mask, definindo os píxeis pretos como 100% transparentes e os brancos como 100% opacos
    nome_ficheiro = f"img_{i}.png" #criar nome do ficheiro
    new_path = os.path.join(ds_A,'train','images',nome_ficheiro) #criar path
    canvas.save(new_path)      
    #Guardar label no documento
    linha = f"{nome_ficheiro} 1 {int(label)} {x} {y} {w} {h}\n"
    f_labels1A.write(linha)

f_labels1A.close()

#TEST
for i in range(NUM_TEST):
    canvas = Image.new('L',(128,128),0)

    idx = random.randint(0,len(mnist_test)-1)

    img_tensor, label_tensor = mnist_test[idx]
    label = torch.argmax(label_tensor).item()
    
    img_np = img_tensor.detach().cpu().numpy().squeeze()

    img_min = img_np.min()
    img_max = img_np.max()
    img_np = (img_np - img_min) / (img_max - img_min + 1e-8) 
    
    img_np[img_np<0.2]=0
    
    img_digito = Image.fromarray((img_np*255).astype(np.uint8), mode='L')

    #Dígitos com o mesmo tamanho 28x28
    s = 28 
    img_digito=img_digito.resize((s,s), Image.Resampling.LANCZOS)
    
    #calcular posição aleatória na imagem preta 
    w, h = img_digito.size 
    x = random.randint(0,128-w) 
    y = random.randint(0,128-h)
    #Colar a imagem na posição x,y
    canvas.paste(img_digito,(x,y), img_digito)

    nome_ficheiro = f"img_{i}.png"
    new_path = os.path.join(ds_A,'test','images',nome_ficheiro)
    canvas.save(new_path)
    #Guardar label
    linha = f"{nome_ficheiro} 1 {int(label)} {x} {y} {w} {h}\n" #o 1 é para a linha em labels.txt ficar no mesmo formato que a versão D
    f_labels2A.write(linha)
    
f_labels2A.close()
print("Criação de Cenas - Versão A concluída")

#VERSÃO D
#Criar imagens com multiplos dígitos (3 a 5) com tamanhos variados e sem sobreposição
#TRAIN
for i in range(NUM_TRAIN):
    canvas = Image.new('L',(128,128),0)
    n = random.randint(3,5) #escolha random do nº de digitos da imagem (entre 3 e 5)
    idx=[]                  #criação de lista vazia para os índices
    labels1D = []           #criação de lista para labels desta imagem
    boxes = []              #lista de posições ocupadas
    for e in range(n):     #para um número n de dígitos
        indice = random.randint(0,len(mnist_train)-1) #escolher um índice random
        idx.append(indice)      #adicionar à lista de indices
    #quando este ciclo for termina, temos uma lista idx com os indices random escolhidos
    
    num = len(idx)      
    for u in range(num):    #para cada digito da lista idx 
        ind = idx[u]        
        img_tensor , label_tensor = mnist_train[ind]
        label = torch.argmax(label_tensor).item()

        img_np = img_tensor.detach().cpu().numpy().squeeze()

        img_min = img_np.min()
        img_max = img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-8) # escala 0 a 1

        img_np[img_np<0.2]=0
        img_digito = Image.fromarray((img_np*255).astype(np.uint8), mode='L')

        s = random.randint(22,36) #num random entre 22 e 36 para a escala
        img_digito=img_digito.resize((s,s), Image.Resampling.LANCZOS)
        
        #calcular posição aleatória na imagem preta 
        w, h = img_digito.size 

        #Perceber se colidiu com alguma imagem no canvas
        colidiu = True
        tentativas = 0

        while colidiu and tentativas<100: #faz 100 tentativas
            x_cand = random.randint(0,128-w) #coordenadas candidatas random
            y_cand = random.randint(0,128-h)
            #definir bordas limite do novo digito
            x_max_cand = x_cand + w
            y_max_cand = y_cand + h 

            sobrepõe = False
            #Se a lista boxes estiver vazia ignora-se este ciclo for
            for (bx_min, by_min, bx_max, by_max) in boxes: 
                #verifica se as coordenadas candidatas colidem com alguma coordenada já existentes
                if not (x_max_cand <= bx_min or x_cand >= bx_max or y_max_cand <= by_min or y_cand >= by_max):
                    sobrepõe = True
                    break
            #se não cumprir as condições considera-se que sobrepõe e sorteiam-se novas coordenadas

            if not sobrepõe:
                # Se não sobrepõe, guarda as coordenadas e sai do while
                x, y = x_cand, y_cand
                boxes.append((x, y, x + w, y + h))
                colidiu = False
            
            tentativas += 1

        #Depois de decidir a coordenada do dígito, colar na imagem a preto
        if not colidiu:
            canvas.paste(img_digito, (x, y), img_digito)
            labels1D.append(f"{label} {x} {y} {w} {h}")


    nome_ficheiro = f"img_{i}.png"
    new_path = os.path.join(ds_D,'train','images',nome_ficheiro)
    canvas.save(new_path)
    
    info_total = " ".join(labels1D)
    f_labels1D.write(f"{nome_ficheiro} {len(labels1D)} {info_total}\n")

f_labels1D.close()

#TEST
for i in range(NUM_TEST):
    canvas = Image.new('L',(128,128),0)
    n = random.randint(3,5) 
    idx=[]  
    labels2D = []
    boxes = [] 
    for e in range(n): 
        indice = random.randint(0,len(mnist_test)-1) 
        idx.append(indice)   
    
    num = len(idx)
    for u in range(num):
        ind = idx[u]
        img_tensor , label_tensor = mnist_test[ind]
        label = torch.argmax(label_tensor).item()

        img_np = img_tensor.detach().cpu().numpy().squeeze()

        img_min = img_np.min()
        img_max = img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-8) 

        img_np[img_np<0.2]=0
    
        img_digito = Image.fromarray((img_np*255).astype(np.uint8), mode='L')

        s = random.randint(22,36) 
        img_digito=img_digito.resize((s,s), Image.Resampling.LANCZOS)
        
        #calcular posição aleatória na imagem preta 
        w, h = img_digito.size 

        colidiu = True
        tentativas = 0

        while colidiu and tentativas<100:
            x_cand = random.randint(0,128-w) 
            y_cand = random.randint(0,128-h)
            #definir bordas limite do novo digito
            x_max_cand = x_cand + w
            y_max_cand = y_cand + y

            sobrepõe = False
            for (bx_min, by_min, bx_max, by_max) in boxes:
                
                if not (x_max_cand <= bx_min or x_cand >= bx_max or y_max_cand <= by_min or y_cand >= by_max):
                    sobrepõe = True
                    break
            
            if not sobrepõe:
                # Se não sobrepõe nada, guarda as coordenadas e sai do while
                x, y = x_cand, y_cand
                boxes.append((x, y, x + w, y + h))
                colidiu = False
            
            tentativas += 1

        if not colidiu:
            canvas.paste(img_digito, (x, y), img_digito)
            labels2D.append(f"{label} {x} {y} {w} {h}")
    

    nome_ficheiro = f"img_{i}.png"
    new_path = os.path.join(ds_D,'test','images',nome_ficheiro)
    canvas.save(new_path)
    #Guardar label
    info_total = " ".join(labels2D)
    f_labels2D.write(f"{nome_ficheiro} {len(labels2D)} {info_total}\n")

f_labels2D.close()
print("Criação de Cenas - Versão D concluída")

