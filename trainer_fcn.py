import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class TrainerFCN:
    def __init__(self, model, train_dataset, test_dataset, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        
        # Criar os Dataloaders internamente para manter o estilo do grupo
        self.train_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=args['batch_size'], shuffle=False
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.get('lr', 0.001))
        
        # Criar pesos: dígitos (0-9) com peso 1.0, fundo (índice 10) com peso 0.1
        weights = torch.ones(11)
        weights[10] = 0.1 # Reduz o peso do fundo para a rede focar nos números
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        # Loss Function para 11 classes (0-9 + fundo)
        #self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for images, targets in tqdm(self.train_loader, desc="Training FCN"):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward: Output shape [Batch, 11, H_out, W_out]
            outputs = self.model(images)
            
            # Se o target_map do dataset for maior que o output da rede, 
            # redimensionamos o output para bater certo com o target.
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                # Calcular precisão pixel a pixel (métrica simples para FCN)
                _, predicted = torch.max(outputs, 1)
                correct_pixels += (predicted == targets).sum().item()
                total_pixels += targets.nelement()
                
        return test_loss / len(self.test_loader), correct_pixels / total_pixels

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Modelo guardado em {path}")