from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F 

class ModelBetterFCN(nn.Module):
    def __init__(self):
        super(ModelBetterFCN, self).__init__()

        #Bloco 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #entra 1 canal saem 32 'lentes'
        self.bn1 = nn.BatchNorm2d(32) #Batch Normalization (ajuda a estabilizar e acelerar o treino)
        self.pool1 = nn.MaxPool2d(2, 2) #olha para píxeis de 2 em 2 e escolhe o valor mais alto
        #128x128 passa a 64x64

        #Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) 
        #64x64 passa a 32x32

        #Bloco 3 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) 
        #Mantém-se 32x32

        # Classificador FCN, em vez de 'achatarmos' a imagem num vetor longo, a rede mantém a info espacial
        self.fc1_conv = nn.Conv2d(128,256, kernel_size=3, padding=1) #256 características complexas
        self.dropout = nn.Dropout(0.5) #no treino, desliga-se 50% dos neurónios , força a rede a ser mais robusta e a não 'decorar'
        
        self.fc2_conv = nn.Conv2d(256, 11,kernel_size=1) #11 classes, classificador pontual 

        print(f'ModelBetterFCN initialized with {self.getNumberOfParameters()} parameters.')
    
    #organização do fluxo
    def forward(self, x):
        # Bloco 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    
        # Bloco 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = F.relu(self.bn3(self.conv3(x)))

        #Camadas Densas convertidas em Convoluções
        x = F.relu(self.fc1_conv(x))
        x = self.dropout(x)
        y = self.fc2_conv(x) #geração final do mapa de 11 camadas
        #devolve um tensor: [batch,11,32,32]
        return y                                  
    
    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)