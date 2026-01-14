
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F  # New import for functional API


class ModelFCN(nn.Module):

    def __init__(self):
        super(ModelFCN, self).__init__() # Chama o construtor da classe pai (nn.Module)

        nrows = 28 # Número de linhas da imagem
        ncols = 28 # Número de colunas da imagem
        ninputs = nrows * ncols # Número de inputs (28x28=784)
        noutputs = 10 # Número de outputs (10 classes para os dígitos 0-9)

        # Define the layers of the model (fully connected layer) 
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.') # Print number of parameters
        summary(self, input_size=(1, 1, 28, 28)) # Print model summary

    def forward(self, x):

        x = x.view(x.size(0), -1) # Esticar o input para um vetor 1D
        y = self.fc(x) # Passar pelo layer fully connected

        return y # Retorna a saída

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # Conta o número de parâmetros treináveis


class ModelConvNet(nn.Module):

    def __init__(self):

        super(ModelConvNet, self).__init__()  # call the parent constructor

        nrows = 28 # Número de linhas da imagem
        ncols = 28 # Número de colunas da imagem
        ninputs = nrows * ncols # Número de inputs (28x28=784)
        noutputs = 10 # Número de outputs (10 classes para os dígitos 0-9)

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define the first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        # print('Forward method called ...')

        # print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        # print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        # print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        # print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        # print('After pool2 x.shape = ' + str(x.shape))

        # Transform to latent vector
        x = x.view(-1, 64*7*7)
        # print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        # print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        # print('Output y.shape = ' + str(y.shape))

        return y


class ModelConvNet3(nn.Module):
    """This is a more complex ConvNet model with 3 conv layers."""

    def __init__(self):

        super(ModelConvNet3, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define second conv layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # this will output ?

        # Define the second pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output ?

        # Define the first fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        # print('Forward method called ...')

        # print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        # print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        # print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        # print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        # print('After pool2 x.shape = ' + str(x.shape))

        x = self.conv3(x)
        # print('After conv3 x.shape = ' + str(x.shape))

        x = self.pool3(x)
        # print('After pool3 x.shape = ' + str(x.shape))

        # Transform to latent vector
        x = x.view(-1, 128*2*2)
        # print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        # print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        # print('Output y.shape = ' + str(y.shape))

        return y

# ------------------------------------------------
# TAREFA 1 - Classe de modelo CNN melhorada com Batch Normalization e Dropout
# ------------------------------------------------

class ModelBetterFCN(nn.Module):
    def __init__(self):
        super(ModelBetterFCN, self).__init__()

        # Arquitetura Melhorada
        
        # Bloco 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # Output: 32x28x28
        self.bn1 = nn.BatchNorm2d(32) # Batch Normalization (ajuda a estabilizar e acelerar o treino)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: 32x14x14

        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Output: 64x14x14
        self.bn2 = nn.BatchNorm2d(64) # Batch Normalization (ajuda a estabilizar e acelerar o treino)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: 64x7x7

        # Bloco 3 (Sem pooling para manter mais informação espacial)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # Output: 128x7x7
        self.bn3 = nn.BatchNorm2d(128) # Batch Normalization (ajuda a estabilizar e acelerar o treino)
        # Output continua 128x7x7

        # Classificador FCN
        #Trocamos o kernel 7 por 3 para ser mais flexível
        self.fc1_conv = nn.Conv2d(128,256, kernel_size=3, padding=1) # Transformar para vetor, Output: 256 neurónios
        self.dropout = nn.Dropout(0.5) # Dropout (50% probabilidade)
        
        self.fc2_conv = nn.Conv2d(256, 11,kernel_size=1) # Output: 10 classes

        #self.classifier = nn.Conv2d(64, 11, kernel_size=1) # 11 = (0 a 9 dígitos + 1 fundo)
        print(f'ModelBetterFCN initialized with {self.getNumberOfParameters()} parameters.')
    #def getNumberOfParameters(self):
        #return sum(p.numel() for p in self.parameters() if p.requires_grad) # Conta o número de parâmetros treináveis

    def forward(self, x):
        # Bloco 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloco 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Camadas Densas convertidas em Convoluções
        x = F.relu(self.fc1_conv(x))
        x = self.dropout(x)
        y = self.fc2_conv(x) 
        
        # O output aqui será [Batch, 10, H, W]
        # Para a Tarefa 4, o H e o W dependerão do tamanho da imagem de entrada!
        return y
    
    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)