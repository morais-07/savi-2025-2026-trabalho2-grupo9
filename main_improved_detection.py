import torch
import os
from model_fcn import ModelBetterFCN
from dataset_fcn import SceneDatasetA, SceneDatasetD
from trainer_fcn import TrainerFCN
import cv2
import numpy as np
import matplotlib.pyplot as plt


#IoU (Intersection over Union)
#Comparação entre a caixa real e a caixa prevista
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    #coordenadas de cruzamento das 2 caixas
    #Cálculo area de interseção
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]   #A prevista
    areaB = boxB[2] * boxB[3]   #A real

    return inter / (areaA + areaB - inter + 1e-6)
#IoU 1.0, sobreposição perfeita
#IoU 0.0, não se tocam


#Testar e métricas
def testar_e_guardar_resultados(model, dataset, device, folder_name="resultados"):
    os.makedirs(folder_name, exist_ok=True)
    model.eval()

    print(f"\nA testar modelo e a gerar resultados em: {folder_name}")

    total_gt = 0
    true_positives = 0
    false_positives = 0

    for i in range(len(dataset)):
        img_tensor, target = dataset[i]

        input_tensor = img_tensor.unsqueeze(0).to(device) #adiciona dimensão de batch

        with torch.no_grad():   #desativa cálculo de gradientes
            output = model(input_tensor)
            output = torch.nn.functional.interpolate(
                output, size=(128, 128), mode='bilinear'
            )                                                   #"esticar" o resultado de volta para 128x128, o output das FCN diminui devido às convoluções 

            probs = torch.nn.functional.softmax(output, dim=1).squeeze()      #transformação de valores brutos em probabilidades
            digit_probs, _ = torch.max(probs[0:10, :, :], dim=0)              #ignorar a classe 10 de fundo e ficar com valor máx
            digit_probs = digit_probs.cpu().numpy()                             #'onde há um dígito'

            pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()      #'qual é o dígito'

        #GT (32x32 → 128x128)
        target_np = target.cpu().numpy().astype(np.uint8)         #preparação de dados, torch - open cv

        target_up = cv2.resize(
            target_np, (128, 128), interpolation=cv2.INTER_NEAREST
        )

        #mapa de classes em coordenadas 
        gt_mask = (target_up < 10).astype(np.uint8)  #tudo o que é <10:True, resto False (fundo)

        contours_gt, _ = cv2.findContours(
            gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )                                                        #identificação de contornos (através de True e False)

        gt_boxes = []
        for cnt in contours_gt:          #para cada um destes, se a área for maior que 50, cria-se uma bounding box
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                gt_boxes.append((x, y, w, h))

        total_gt += len(gt_boxes)        #foi identificado um dígito

        #predição de onde há dígito baseado no modelo FCN
        heatmap_gray = (digit_probs * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET) #coloca resultado da fcn em algo visível, criação do mapa de calor

        img_out = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)   #preparação da imagem original
        img_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        threshold = 0.6                                                     #só se considera que há sígito se a rede tiver mais que 60% de certeza
        mask = (digit_probs > threshold).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(                                     #tal como para o gt, são procurados contornos e adicionados à lista de de caixas previstas com FCN
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        pred_boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 80:
                x, y, w, h = cv2.boundingRect(cnt)
                pred_boxes.append((x, y, w, h))

                roi = pred_map[y:y+h, x:x+w]                                #desenha no pred_map (onde há info sobre classes), desenha bounding box baseada na previsão da FCN
                roi_digits = roi[roi < 10]

                if len(roi_digits) > 0:                                     #desenhar bounding box
                    cls = np.argmax(np.bincount(roi_digits))
                    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(
                        img_color, str(cls), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

        #matching IoU
        matched_gt = set()

        for pb in pred_boxes:                                 #para cada caixa prevista com FCN, se já tiver match com gt, ignora, se não tiver e a sobreposição (IoU) for > 0.5: TP
            matched = False                                     #else: FP
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if iou(pb, gt) > 0.5:
                    true_positives += 1
                    matched_gt.add(j)
                    matched = True
                    break
            if not matched:
                false_positives += 1

        #guardar algumas imagens
        if i < 5:
            img_orig = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
            comparison = np.hstack((img_orig, heatmap_color, img_color))
            cv2.imwrite(f"{folder_name}/resultado_completo_{i}.png", comparison)
            cv2.imwrite(f"{folder_name}/resultado_{i}.png", img_color)

    #métricas finais
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (total_gt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n--- TABELA DE RESULTADOS ---")
    print(f"Total GT        | {total_gt}")
    print(f"True Positives  | {true_positives}")
    print(f"False Positives | {false_positives}")
    print(f"Precision       | {precision*100:.2f}%")
    print(f"Recall          | {recall*100:.2f}%")
    print(f"F1-Score        | {f1:.4f}")



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
        print("\n" + "=" * 50)
        print(f"INICIANDO TREINO: VERSÃO {version}")
        print("=" * 50)

        if version == 'A':
            root_dir = 'Dataset_Cenas_Versão_A'
            train_data = SceneDatasetA(root_dir, 'train', args['output_size'])
            test_data = SceneDatasetA(root_dir, 'test', args['output_size'])
        else:
            root_dir = 'Dataset_Cenas_Versão_D'
            train_data = SceneDatasetD(root_dir, 'train', args['output_size'])
            test_data = SceneDatasetD(root_dir, 'test', args['output_size'])

        model = ModelBetterFCN().to(args['device'])
        trainer = TrainerFCN(model, train_data, test_data, args)

        for epoch in range(args['epochs']):
            train_loss = trainer.train_epoch()
            test_loss, pixel_acc = trainer.evaluate()

            print(
                f"Versão {version} | Epoch {epoch+1:02d}/{args['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Pixel Acc: {pixel_acc*100:.2f}%"
            )

        save_path = f"model_fcn_version_{version}.pth"
        trainer.save_checkpoint(save_path)
        print(f"Modelo guardado em: {save_path}")

        testar_e_guardar_resultados(
            model, test_data, args['device'],
            folder_name=f"resultados_ver_{version}"
        )

    print("\n" + "!" * 50)
    print("TODOS OS TREINOS E TESTES CONCLUÍDOS COM SUCESSO!")
    print("!" * 50)


if __name__ == "__main__":
    main()
