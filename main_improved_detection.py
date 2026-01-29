import torch
import os
from model_fcn import ModelBetterFCN
from dataset_fcn import SceneDatasetA, SceneDatasetD
from trainer_fcn import TrainerFCN
import cv2
import numpy as np
import matplotlib.pyplot as plt

#função auxiliar 
def calculate_iou(boxA, boxB):
    # Calcula a interseção (zona comum)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Áreas individuais
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # União = Área A + Área B - Interseção
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

#função métricas
def calcular_tabela_metricas(model, dataset, device):
    model.eval()
    tp, fp, total_gt = 0, 0, 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, target_mask, _ = dataset[i]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            #Predição da rede
            output_cls, output_reg = model(input_tensor)

            output_cls = torch.nn.functional.interpolate(output_cls, size=(128, 128), mode='bilinear')
            probs = torch.nn.functional.softmax(output_cls, dim=1).squeeze()
            digit_probs, _ = torch.max(probs[0:10, :, :], dim=0)
            pred_map = torch.argmax(output_cls, dim=1).squeeze().cpu().numpy()

            #redimensionar target_mask para 128x128
            target_mask_np = target_mask.numpy().astype(np.uint8)
            target_mask_full = cv2.resize(target_mask_np, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            #Ground Truth
            gt_mask = (target_mask_full < 10).astype(np.uint8)
            gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_boxes = []
            for cnt in gt_contours:
                if cv2.contourArea(cnt) > 100: # Filtro de área mínima para GT real em 128x128
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Extrair a classe do mapa redimensionado para evitar erros de índice
                    roi_gt = target_mask_full[y:y+h, x:x+w]
                    #Ignorar fundo (10)
                    real_pixels = roi_gt[roi_gt < 10]
                    if len(real_pixels) > 0:
                        real_cls = np.bincount(real_pixels).argmax()
                        gt_boxes.append({'box': (x,y,w,h), 'class': real_cls, 'matched': False})
            
            total_gt += len(gt_boxes)

            #Caixas previstas pelo modelo
            # Usamos threshold de 0.7 e morfologia OPEN 5x5 para limpar falsos positivos
            mask_binaria = (digit_probs.cpu().numpy() > 0.7).astype(np.uint8)
            mask_cleaned = cv2.morphologyEx(mask_binaria, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            pred_contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for p_cnt in pred_contours:
                if cv2.contourArea(p_cnt) > 250:
                    M = cv2.moments(p_cnt)
                    if M["m00"] == 0: continue
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    
                    # Extraímos px, py, pw, ph da CABEÇA DE REGRESSÃO (para provar que funciona)
                    rv = output_reg[:, cY, cX]
                    px, py, pw, ph = int(rv[0]*128), int(rv[1]*128), int(rv[2]*128), int(rv[3]*128)
                    
                    roi_pred = pred_map[max(0,cY-2):min(128,cY+2), max(0,cX-2):min(128,cX+2)]
                    roi_digits = roi_pred[roi_pred < 10]
                    if len(roi_digits) == 0: continue
                    pred_cls = np.bincount(roi_digits.flatten()).argmax()
                    
                    match_found = False
                    for gt in gt_boxes:
                        iou = calculate_iou((px, py, pw, ph), gt['box'])
                        if iou > 0.35 and pred_cls == gt['class'] and not gt['matched']:
                            tp += 1
                            gt['matched'] = True
                            match_found = True
                            break
                    if not match_found: fp += 1

    #resultados finais
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_gt, tp, fp, precision, recall, f1

#transformar mapa de calor da rede em bounding boxes
def testar_e_guardar_resultados(model, dataset, device, folder_name="resultados"):
    os.makedirs(folder_name, exist_ok=True)
    model.eval() #garante que a rede não desliga 50%
    print(f"A gerar visualizações em: {folder_name}...")
    
    # Testar apenas 5 imagens
    for i in range(5):
        img_tensor, target = dataset[i] #Vai buscar ao dataset
        #img_tensor tem shape [1, 128, 128]
        
        input_tensor = img_tensor.unsqueeze(0).to(device) #adiciona o batch
        
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.nn.functional.interpolate(output, size=(128, 128), mode='bilinear')
            #Gerar Heat Map
            #Gerar probabilidades para o Heatmap (usando Softmax)
            probs = torch.nn.functional.softmax(output, dim=1).squeeze()
            #Pegamos na probabilidade máxima entre todas as classes de dígitos (0 a 9)
            #Ignoramos a classe de fundo (assumindo que é a última, índice 10)
            digit_probs, _ = torch.max(probs[0:10, :, :], dim=0)  #matriz 2D onde cada ponto brilha mais se a rede tiver > certeza que há um dígito
            digit_probs = digit_probs.cpu().numpy()

            pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy() 

        #Processamento do heat map
        heatmap_gray = (digit_probs * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

        #Converter para imagem colorida para desenhar
        img_out = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        
        threshold = 0.6 #ajustar entre 0.5 e 0.8
        mask_all_digits = (digit_probs > threshold).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8) #serve para limpezas 'finas'
        mask_cleaned = cv2.morphologyEx(mask_all_digits, cv2.MORPH_OPEN, kernel) 
            
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #identificação de contornos externos
        for cnt in contours:
            if cv2.contourArea(cnt) > 80:   #para descartar falsos positivos
                x, y, w, h = cv2.boundingRect(cnt) #retângulo que envolve o contorno
        
                #Extrair a região correspondente no pred map
                roi = pred_map[y:y+h, x:x+w]
                #Filtrar apenas os pixels que não são fundo na ROI
                roi_digits = roi[roi < 10]
        
                if len(roi_digits) > 0:
                    #Encontrar a classe mais comum nessa zona
                    counts = np.bincount(roi_digits)
                    final_class = np.argmax(counts)
            
                    #Desenhar apenas uma caixa para o objeto inteiro
                    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img_color, str(final_class) , (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        #Criar comparação lado a lado (Original | Heatmap | Deteção Final)
        img_orig_bgr = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        comparison = np.hstack((img_orig_bgr, heatmap_color, img_color))

        #Guardar resultado
        cv2.imwrite(f"{folder_name}/resultado_completo_{i}.png", comparison)
        
        cv2.imwrite(f"{folder_name}/resultado_{i}.png", img_color)


def main():
    VERSIONS_TO_TRAIN = ['A', 'D']
    
    args = {
        'batch_size': 16,
        'lr': 0.001,
        'epochs': 15,
        'output_size': (32, 32), 
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    for version in VERSIONS_TO_TRAIN:
        print(f"\n" + "="*50)
        print(f"INICIANDO TREINO: VERSÃO {version}")
        print(f"="*50)

        if version == 'A':
            root_dir = 'Dataset_Cenas_Versão_A'
            train_data = SceneDatasetA(root_dir, subset='train', output_size=args['output_size'])
            test_data = SceneDatasetA(root_dir, subset='test', output_size=args['output_size'])
        else:
            root_dir = 'Dataset_Cenas_Versão_D'
            train_data = SceneDatasetD(root_dir, subset='train', output_size=args['output_size'])
            test_data = SceneDatasetD(root_dir, subset='test', output_size=args['output_size'])

        
        model = ModelBetterFCN() 
        trainer = TrainerFCN(model, train_data, test_data, args)

        for epoch in range(args['epochs']):
            train_loss = trainer.train_epoch()
            test_loss, pixel_acc = trainer.evaluate()
            
            print(f"Versão {version} | Epoch {epoch+1:02d}/{args['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Pixel Acc: {pixel_acc*100:.2f}%")


        save_path = f"model_fcn_version_{version}.pth"
        trainer.save_checkpoint(save_path)
        print(f"Finalizado: Modelo Versão {version} guardado.")
        print(f"A testar modelo da Versão {version}...")
        testar_e_guardar_resultados(model, test_data, args['device'], folder_name=f"resultados_ver_{version}")
        total_gt, tp, fp, prec, rec, f1 = calcular_tabela_metricas(model, test_data, args['device'])
    
        print(f"\n--- TABELA DE RESULTADOS: VERSÃO {version} ---")
        print(f"Total GT | True Positives | False Positives | Precision | Recall | F1-Score")
        print(f"{total_gt:8d} | {tp:14d} | {fp:15d} | {prec:.2%} | {rec:.2%} | {f1:.4f}")

    print("\n" + "!"*50)
    print("TODOS OS TREINOS E TESTES CONCLUÍDOS COM SUCESSO!")
    print("!"*50)

if __name__ == "__main__":
    main()
