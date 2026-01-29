import torch

#torch.backends.nnpack.enabled = False
import sys
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import nms 
from PIL import Image, ImageDraw
import os
import glob
import json # Adicionado para manipulação de ficheiros JSON
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tarefa_1.model import ModelBetterCNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report # Adicionado para gerar o relatório detalhado

def load_trained_model(checkpoint_path, device):

    # ---------------------------
    # Importa o modelo betterCNN
    # ---------------------------

    model = ModelBetterCNN().to(device) # Importa o modelo betterCNN
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    print(f"[INFO] A carregar modelo de: {checkpoint_path}")
    
    # ---------------------------
    # Tenta carregar os pesos do modelo
    # ---------------------------
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # Carrega o checkpoint
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: # Verifica se é um dict com os pesos
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict): 
            # Tenta carregar direto se o dict forem só os pesos
            model.load_state_dict(checkpoint, strict=False)
        else:
            # Caso seja o modelo inteiro salvo (menos comum hoje em dia)
            model = checkpoint
            
    except Exception as e:
        print(f"[ERRO] Falha ao carregar pesos: {e}")
        exit()

    model.eval() # Coloca o modelo em modo de avaliação
    return model # Retorna o modelo carregado

def calculate_entropy(probs):
    """Calcula a entropia da distribuição de probabilidades (Incerteza)"""
    # Entropia = - Σ p(x) * log(p(x))
    # Se o modelo estiver confuso, a entropia será alta.
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=1) # Entropia por amostra
    return entropy.item()

def sliding_window(image, step_size, window_size):
    """Gera janelas deslizantes sobre a imagem."""

    w, h = image.size
    for y in range(0, h - window_size[1] + 1, step_size): # O '+1' garante que a última janela é incluída
        for x in range(0, w - window_size[0] + 1, step_size): # O '+1' garante que a última janela é incluída
            window = image.crop((x, y, x + window_size[0], y + window_size[1])) # (left, upper, right, lower)
            yield (x, y, window) # Retorna a posição (x, y) e a janela

def calculate_iou(box1, box2):
    """Calcula se duas caixas se sobrepõem (Intersection over Union)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1) # Área de interseção
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # Área da caixa 1
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # Área da caixa 2
    
    union = area1 + area2 - intersection # Área de união
    return intersection / union if union > 0 else 0 # Retorna IoU

def visualize_grid(images_list, titles_list):
    """Cria um mosaico (grelha) de até 3x3 com as imagens de resultado"""
    if not images_list: 
        print("[AVISO] Nenhuma imagem para mostrar na grelha.")
        return
    
    # Limita a grelha a no máximo 9 imagens (3x3)
    num_imgs = min(len(images_list), 9)
    images_list = images_list[:num_imgs]
    titles_list = titles_list[:num_imgs]
    
    # Define o tamanho da figura
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    # Preenche a grelha
    for i in range(9):
        if i < num_imgs:
            axes[i].imshow(images_list[i])
            axes[i].set_title(titles_list[i], fontsize=8)
        axes[i].axis('off') # Esconde os eixos mesmo nas células vazias
        
    plt.tight_layout() # Ajusta o layout
    plt.show() # Mostra a grelha

def load_ground_truth(labels_path):
    """Lê o ficheiro labels.txt. Formato: [Nome] [N_Digitos] [L X Y W H] ..."""

    gt_data = {} # Dicionário para armazenar os dados de ground truth
    if not os.path.exists(labels_path):
        return {}
    
    with open(labels_path, 'r') as f: # Lê o ficheiro linha a linha
        for line in f: 
            parts = line.strip().split() # Divide a linha em partes
            if len(parts) < 2: continue # Linha inválida
            filename = parts[0] # Nome do ficheiro de imagem
            boxes = []
            
            # Número de dígitos na imagem
            num_digits = int(parts[1])
            cursor = 2 # Posição inicial dos dados dos dígitos
            
            # Lê os dados de cada dígito
            for _ in range(num_digits):
                try:
                    # L, X, Y, W, H
                    lbl, x, y, w, h = map(int, parts[cursor:cursor+5])
                    boxes.append([x, y, x+w, y+h, lbl]) # Convertemos para [x1, y1, x2, y2, label]
                    cursor += 5
                except: break
            gt_data[filename] = boxes # Armazena as caixas no dicionário
    return gt_data # Retorna o dicionário com os dados de ground truth

def process_version(config, model, device, transform, limit=None):
    """Processa uma versão específica do dataset."""

    print(f"\n>>> A PROCESSAR: {config['name']}")
    
    output_dir = config['output_dir'] # Diretório para guardar os resultados
    os.makedirs(output_dir, exist_ok=True)

    # Carrega os dados de ground truth

    ground_truth = load_ground_truth(config['labels_file'])
    image_files = glob.glob(os.path.join(config['images_dir'], "*.png"))
    if limit: image_files = image_files[:limit]

    # Métricas acumuladas

    total_tp, total_fp, total_gt = 0, 0, 0
    grid_images, grid_titles = [], []
    
    # Listas para armazenar predições e valores reais para o classification_report
    y_true_all = []
    y_pred_all = []

    # ----------------------------
    # Parâmetros de deteção
    # ----------------------------

    WINDOW_SIZE = (32, 32) # Tamanho da janela deslizante 
    STEP_SIZE = 4 # Passo da janela deslizante ou seja quantos pixels a janela avança a cada passo
    CONF_THRESH = 0.91 # Limiar de confiança para aceitar uma deteção ou seja a probabilidade mínima para considerar que a predição é válida
    ENTROPY_THRESH = 0.1 # Limiar de entropia para aceitar uma deteção ou seja o máximo de incerteza permitido
    IOU_THRESH = 0.05 # Limiar de IoU para Non-Maximum Suppression (NMS) ou seja o máximo de sobreposição permitido entre caixas para serem consideradas distintas
    PIXEL_THRESHOLD = 0.03 # Limiar de pixel para ignorar janelas quase vazias

    # ----------------------------
    # Processar cada imagem
    # ----------------------------

    for img_path in tqdm(image_files, desc=f"Scanning {config['name']}", leave=False): # Para cada imagem no diretório
        filename = os.path.basename(img_path)
        all_boxes, all_scores, all_labels = [], [], []
        
        img = Image.open(img_path).convert('L') # Abre a imagem em escala de cinza
        draw_img = img.convert('RGB') # Para desenhar as caixas
        draw = ImageDraw.Draw(draw_img) # Objeto para desenhar na imagem

        # ----------------------------
        # Sliding window
        # ----------------------------

        for (x, y, window) in sliding_window(img, STEP_SIZE, WINDOW_SIZE):
            if transforms.ToTensor()(window).mean() < PIXEL_THRESHOLD: # Ignora janelas quase vazias
                continue 

            input_tensor = transform(window).unsqueeze(0).to(device) # Prepara o tensor de entrada
            with torch.no_grad(): # Sem gradientes
                output = model(input_tensor) # Forward pass
                probs = F.softmax(output, dim=1) # Probabilidades por classe
                conf, pred = torch.max(probs, 1) # Confiança e predição da classe
                entropy = calculate_entropy(probs) # Calcula a entropia
            
            if conf.item() >= CONF_THRESH and entropy < ENTROPY_THRESH: # Filtra por confiança e entropia
                all_boxes.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]]) # Caixa [x1, y1, x2, y2]
                all_scores.append(conf.item()) # Confiança
                all_labels.append(pred.item()) # Classe prevista

        final_preds = [] # Lista de predições finais após NMS


        if all_boxes:
            keep = nms(torch.tensor(all_boxes, dtype=torch.float32), 
                       torch.tensor(all_scores, dtype=torch.float32), IOU_THRESH)
            for idx in keep:
                idx = idx.item()
                final_preds.append((all_boxes[idx], all_labels[idx]))
                draw.rectangle(all_boxes[idx], outline="red", width=1)
                draw.text((all_boxes[idx][0], all_boxes[idx][1]-10), str(all_labels[idx]), fill="red")

        # -------------------
        # Avaliação
        # -------------------

        if filename in ground_truth:
            gt_boxes = ground_truth[filename] # Caixas de ground truth
            total_gt += len(gt_boxes) # Atualiza o total de dígitos reais
            matched_gt = [False] * len(gt_boxes) # Marcações de dígitos encontrados para métricas globais
            
            # Lista auxiliar para controlar quais predições já foram "gastas" no alinhamento do JSON
            temp_matched_preds = [False] * len(final_preds)

            # 1. Loop para calcular TP e FP (Estatísticas globais do terminal)
            for p_idx, (p_box, p_lbl) in enumerate(final_preds):  # Para cada previsão
                found = False  
                for i, g_box in enumerate(gt_boxes):  # Para cada ground truth
                    if p_lbl == g_box[4] and calculate_iou(p_box, g_box[:4]) > 0.3:   # Verifica se a predição corresponde ao ground truth
                         if not matched_gt[i]: # Se o GT ainda não foi correspondido
                            matched_gt[i] = True   # Marca o GT como correspondido
                            total_tp += 1 # Incrementa os verdadeiros positivos
                            found = True # Marca que encontrou um match
                            break
                if not found: total_fp += 1 # Incrementa os falsos positivos se não encontrou match

            # 2. Loop para preencher y_true e y_pred (Estatísticas detalhadas para o JSON)
            # Alinhamos cada Ground Truth com uma predição ou marcamos como falha (-1)
            for i, g_box in enumerate(gt_boxes): # Para cada ground truth
                g_lbl = g_box[4] # Label do ground truth
                y_true_all.append(g_lbl) # Valor Real
                
                match_idx = -1 
                for j, (p_box, p_lbl) in enumerate(final_preds): # Para cada previsão
                    if not temp_matched_preds[j] and p_lbl == g_lbl and calculate_iou(p_box, g_box[:4]) > 0.3: # Verifica se a predição corresponde ao ground truth
                        match_idx = j # Índice da predição correspondente
                        break
                
                if match_idx != -1: # Encontrou uma predição correspondente
                    y_pred_all.append(final_preds[match_idx][1]) # Valor Previsto (Correto)
                    temp_matched_preds[match_idx] = True
                else:
                    y_pred_all.append(-1) # Falso Negativo (O modelo não viu o dígito)

            # Adicionamos as predições que sobraram como Falsos Positivos
            for j, (p_box, p_lbl) in enumerate(final_preds):
                if not temp_matched_preds[j]:
                    y_true_all.append(-1) # Não havia dígito real aqui
                    y_pred_all.append(p_lbl) # O modelo inventou uma predição

        draw_img.save(os.path.join(output_dir, f"res_{filename}")) # Guarda a imagem com as caixas desenhadas

        if len(grid_images) < 9: # Limita a 9 imagens para a grelha
            grid_images.append(np.array(draw_img)) # Adiciona a imagem à lista
            grid_titles.append(f"{filename}\nDet: {len(final_preds)}") # Título com número de deteções

    # -----------------------------------------
    # Gerar e Guardar estatísticas em JSON
    # -----------------------------------------
    
    # Identifica o sufixo (A ou D) com base no nome da tarefa
    suffix = "A" if "A" in config['name'] else "D"
    
    # Filtramos para obter apenas as classes dos dígitos (0-9), ignorando o background (-1)
    unique_labels = sorted(list(set([y for y in y_true_all if y != -1] + [y for y in y_pred_all if y != -1])))
    
    # Gerar o dicionário do relatório de classificação
    report_dict = classification_report(y_true_all, y_pred_all, labels=unique_labels, output_dict=True, zero_division=0)
    
    # Define o caminho final: ex: JSON_A/statistics_A.json
    json_path = os.path.join(f'./results_sliding_window/statistics_{suffix}.json')
    
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    
    print(f"[INFO] Estatísticas guardadas em {json_path}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0 # Precisão
    recall = total_tp / total_gt if total_gt > 0 else 0 # Recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # F1-Score
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "gt": total_gt, "imgs": grid_images, "titles": grid_titles}

def main():

    # ------------------------------
    # Carregar o modelo treinado
    # ------------------------------

    CHECKPOINT_PATH = '../experiments/checkpoint.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(CHECKPOINT_PATH, device) # Carrega o modelo treinado
    
    transform = transforms.Compose([ # Transformações para o modelo
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    LIMITAR_TESTE = 100 # Definir para um número para limitar o número de imagens testadas

    # ------------------------------
    # Definir tarefas para cada versão
    # ------------------------------

    tasks = [
        {"name": "Versão A", "images_dir": "../Dataset_Cenas_Versão_A/test/images/", "labels_file": "../Dataset_Cenas_Versão_A/test/labels.txt", "output_dir": "./results_sliding_window/Versao_A"},
        {"name": "Versão D", "images_dir": "../Dataset_Cenas_Versão_D/test/images/", "labels_file": "../Dataset_Cenas_Versão_D/test/labels.txt", "output_dir": "./results_sliding_window/Versao_D"}
    ]

    # ------------------------------
    # Processar cada versão
    # ------------------------------

    results_summary = []
    for task in tasks:
        res = process_version(task, model, device, transform, limit=LIMITAR_TESTE) # Processa a versão
        results_summary.append((task['name'], res)) # Armazena os resultados

    # ------------------------------
    # RELATÓRIO FINAL
    # ------------------------------
    
    print("\n" + "="*45)
    print("        RELATÓRIO FINAL DE MÉTRICAS")
    print("="*45)
    for name, res in results_summary:
        print(f"\n>>> {name.upper()}:")
        print(f"  - Total Real (GT):     {res['gt']}")
        print(f"  - Acertos (TP):        {res['tp']}")
        print(f"  - Erros (FP):           {res['fp']}")
        print(f"  - Precisão:            {res['precision']:.2%}")
        print(f"  - Recall:              {res['recall']:.2%}")
        print(f"  - F1-Score:            {res['f1']:.4f}")
    print("\n" + "="*45)

    for name, res in results_summary:
        visualize_grid(res['imgs'], res['titles']) # Mostra a grelha de imagens

if __name__ == '__main__':
    main()