import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
from colorama import Fore, Style
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model
        # Setup device and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        # For testing we typically set shuffle to false

        # Setup loss function
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        # Start from scratch or resume training
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):

        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        # -----------------------------------------
        # Iterate all epochs
        # -----------------------------------------
        for i in range(self.epoch_idx, self.args['num_epochs']):  # number of epochs

            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))
            # -----------------------------------------
            # Train - Iterate over batches
            # -----------------------------------------
            self.model.train()  # set model to training mode
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                image_tensor = image_tensor.to(self.device)
                label_gt_tensor = label_gt_tensor.to(self.device)

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()  # the actual backpropagation
                self.optimizer.step()

            # -----------------------------------------
            # Test - Iterate over batches
            # -----------------------------------------
            self.model.eval()  # set model to evaluation mode

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                #MUDEI
                image_tensor = image_tensor.to(self.device)
                label_gt_tensor = label_gt_tensor.to(self.device)
                
                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # During test there is no model update

            # ---------------------------------
            # End of the epoch training
            # ---------------------------------
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))
            # print('batch_losses: ' + str(batch_losses))

            # update the training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            # update the testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            # Draw the updated training figure
            self.draw()

            # Save the training state
            self.saveTrain()

        print('Training completed.')
        print('Training losses: ' + str(self.train_epoch_losses))
        print('Test losses: ' + str(self.test_epoch_losses))

    def loadTrain(self):
        print('Resuming training from last checkpoint.')

        # find the checkpoint file
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        print('checkpoint_file: ' + str(checkpoint_file))

        # Verify if file exists. If not abort. Cannot resume without the checkpoint.pkl
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print(checkpoint.keys())

        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])  # contains the model's weights
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])  # contains the optimizer's

    def saveTrain(self):

        # Create the dictionary to save the checkpoint.pkl
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses

        checkpoint['model_state_dict'] = self.model.state_dict()  # contains the model's weights
        # contains the optimizer's state
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        # Save the best.pkl
        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    def draw(self):

        plt.figure(1)  # creates a new fig therefore clears all past drawings
        plt.clf()

        # Setup the figure
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.1])  # type: ignore

        # plot training
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        # plot testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        # draw best checkpoint
        best_epoch_idx = int(np.argmin(self.test_epoch_losses))
        print('best_epoch_idx: ' + str(best_epoch_idx))
        plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)

        plt.legend(['Train', 'Test', 'Best'], loc='upper right')

        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))


# New Evaluate method with sklearn metrics (Tarefa 1.4)

    def evaluate(self):
        
        # -----------------------------------------
        # Iterate over test batches and compute the ground trutch and predicted  values for all examples
        # -----------------------------------------
        self.model.eval()
        num_batches = len(self.test_dataloader)

        # Listas para guardar resultados (Inteiros: 0-9)
        gt_classes = [] 
        predicted_classes = []

        print('A iniciar avaliação...')
        with torch.no_grad(): # Desliga gradientes para poupar memória
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):
                #MUDEI AQUI
                image_tensor = image_tensor.to(self.device)
                label_gt_tensor = label_gt_tensor.to(self.device)
                # Converter One-Hot (do MSELoss) para Inteiro (ex: [0,0,1,0] -> 2)
                batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

                # Previsão
                label_pred_tensor = self.model.forward(image_tensor)
                # Softmax para probabilidades
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
                # Argmax para obter a classe prevista
                batch_predicted_classes = label_pred_probabilities_tensor.argmax(dim=1).tolist()

                gt_classes.extend(batch_gt_classes)
                predicted_classes.extend(batch_predicted_classes)

        # -----------------------------------------
        # TAREFA 1: Avaliação com Sklearn (NOVO)
        # -----------------------------------------
        

        # -----------------------------------------
        # Relatório de Classificação (Precision, Recall, F1-Score)
        # -----------------------------------------
        print("\n" + "="*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO (Precision, Recall, F1)")
        print("="*60)
        print(classification_report(gt_classes, predicted_classes, digits=4)) # digits = x, para x casas decimais

        # -----------------------------------------
        # Desenhar a Matriz de Confusão
        # -----------------------------------------
        cm = confusion_matrix(gt_classes, predicted_classes)
        plt.figure(2, figsize=(10, 8)) # Cria a figura 
        class_names = [str(i) for i in range(10)] # Classes 0-9
        
        seaborn.heatmap(cm, 
                        annot=True, # Anotar células com valores
                        fmt='d', # Formato inteiro
                        cmap='Blues', # Cor azul
                        cbar=False, # Sem barra de cores
                        xticklabels=class_names, # Rótulos do eixo x
                        yticklabels=class_names) # Rótulos do eixo y

        plt.title('Matriz de Confusão', fontsize=16)
        plt.xlabel('Classe Prevista', fontsize=14) 
        plt.ylabel('Classe Real', fontsize=14) 
        plt.tight_layout() # Ajustar o layout

        save_path = os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png') # Guardar figura
        plt.savefig(save_path)
        print(f"Matriz de confusão guardada em: {save_path}")

        # -----------------------------------------
        # Guardar estatísticas em JSON
        # -----------------------------------------

        report_dict = classification_report(gt_classes, predicted_classes, output_dict=True) # output_dict=True para obter dicionário
        
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(report_dict, f, indent=4)
        
        print("Estatísticas guardadas em statistics.json")
