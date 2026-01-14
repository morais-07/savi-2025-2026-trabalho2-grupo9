import torch
import os
from model_fcn import ModelBetterFCN
from dataset_fcn import SceneDatasetA, SceneDatasetD
from trainer_fcn import TrainerFCN
import cv2
import numpy as np
import matplotlib.pyplot as plt


def testar_e_guardar_resultados(model, dataset, device, folder_name="resultados"):
    os.makedirs(folder_name, exist_ok=True)
    model.eval()
    print(f"A gerar visualizações em: {folder_name}...")
    
    # Testar apenas 5 imagens para não demorar muito
    for i in range(5):
        img_tensor, target = dataset[i] # Vai buscar ao dataset
        # img_tensor tem shape [1, 128, 128]
        
        input_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim -> [1, 1, 128, 128]
        
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.nn.functional.interpolate(output, size=(128, 128), mode='bilinear')
            #Gerar Heat Map
            # Gerar probabilidades para o Heatmap (usando Softmax)
            probs = torch.nn.functional.softmax(output, dim=1).squeeze()
            # Pegamos na probabilidade máxima entre todas as classes de dígitos (0 a 9)
            # Ignoramos a classe de fundo (assumindo que é a última, índice 10)
            digit_probs, _ = torch.max(probs[0:10, :, :], dim=0)
            digit_probs = digit_probs.cpu().numpy()

            pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 3. Processamento do Heatmap para visualização
        heatmap_gray = (digit_probs * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

        # Converter para imagem colorida para desenhar
        img_out = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        # Desenhar caixas (Lógica OpenCV)
        for classe in range(10):
            mask = (pred_map == classe).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 30:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(img_color, str(classe), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 5. Criar comparação lado a lado (Original | Heatmap | Deteção Final)
        img_orig_bgr = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        comparison = np.hstack((img_orig_bgr, heatmap_color, img_color))

        # Guardar resultado
        cv2.imwrite(f"{folder_name}/resultado_completo_{i}.png", comparison)
        
        cv2.imwrite(f"{folder_name}/resultado_{i}.png", img_color)


def main():
    # --- 1. CONFIGURAÇÕES GERAIS ---
    # Define as versões que queres treinar
    VERSIONS_TO_TRAIN = ['A', 'D']
    
    args = {
        'batch_size': 16,
        'lr': 0.001,
        'epochs': 15,
        'output_size': (32, 32), # Confirmar se a tua FCN reduz 128 -> 32
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Loop principal que corre as duas versões
    for version in VERSIONS_TO_TRAIN:
        print(f"\n" + "="*50)
        print(f"INICIANDO TREINO: VERSÃO {version}")
        print(f"="*50)

        # --- 2. SELEÇÃO DO DATASET ---
        if version == 'A':
            root_dir = 'Dataset_Cenas_Versão_A'
            train_data = SceneDatasetA(root_dir, subset='train', output_size=args['output_size'])
            test_data = SceneDatasetA(root_dir, subset='test', output_size=args['output_size'])
        else:
            root_dir = 'Dataset_Cenas_Versão_D'
            train_data = SceneDatasetD(root_dir, subset='train', output_size=args['output_size'])
            test_data = SceneDatasetD(root_dir, subset='test', output_size=args['output_size'])

        # --- 3. INICIALIZAÇÃO DO MODELO ---
        # Criamos um modelo novo em cada loop para não "contaminar" o treino
        model = ModelBetterFCN() 
        trainer = TrainerFCN(model, train_data, test_data, args)

        # --- 4. CICLO DE TREINO ---
        for epoch in range(args['epochs']):
            train_loss = trainer.train_epoch()
            test_loss, pixel_acc = trainer.evaluate()
            
            print(f"Versão {version} | Epoch {epoch+1}/{args['epochs']} | "
                  f"Loss: {train_loss:.4f} | Pixel Acc: {pixel_acc*100:.2f}%")

        # --- 5. GUARDAR O MODELO ESPECÍFICO ---
        save_path = f"model_fcn_version_{version}.pth"
        trainer.save_checkpoint(save_path)
        print(f"Finalizado: Modelo Versão {version} guardado.")
        print(f"A testar modelo da Versão {version}...")
        testar_e_guardar_resultados(model, test_data, args['device'], folder_name=f"resultados_ver_{version}")

    print("\n" + "!"*50)
    print("TODOS OS TREINOS CONCLUÍDOS COM SUCESSO!")
    print("!"*50)

if __name__ == "__main__":
    main()
