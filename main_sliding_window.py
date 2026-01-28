import torch

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import nms 
from PIL import Image, ImageDraw
import os
import glob
from tqdm import tqdm
from model import ModelBetterCNN 
import numpy as np
import matplotlib.pyplot as plt

def load_trained_model(checkpoint_path, device):

    model = ModelBetterCNN().to(device) # Importa o modelo betterCNN
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    print(f"[INFO] A carregar modelo de: {checkpoint_path}")
    
    # Carregar pesos
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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

    model.eval()
    return model

def calculate_entropy(probs):
    # Calcula a entropia da distribuição de probabilidades (Incerteza).
    # Entropia = - Σ p(x) * log(p(x))
    # Se o modelo estiver confuso, a entropia será alta.
    log_probs = torch.log(probs + 1e-10) 
    entropy = -torch.sum(probs * log_probs, dim=1) # Entropia por amostra
    return entropy.item()

def sliding_window(image, step_size, window_size):
    w, h = image.size
    # O '+1' garante que chegamos à borda
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            window = image.crop((x, y, x + window_size[0], y + window_size[1]))
            yield (x, y, window)

def calculate_iou(box1, box2):
    """Calcula se duas caixas se sobrepõem (Intersection over Union)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def visualize_grid(images_list, titles_list):
    """Cria um mosaico (grelha) de até 3x3 com as imagens de resultado"""
    if not images_list: 
        print("[AVISO] Nenhuma imagem para mostrar na grelha.")
        return
        
    # Limita a grelha a no máximo 9 imagens (3x3)
    num_imgs = min(len(images_list), 9)
    images_list = images_list[:num_imgs]
    titles_list = titles_list[:num_imgs]
    
    # Define o tamanho da figura (ajusta conforme necessário)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(9):
        if i < num_imgs:
            axes[i].imshow(images_list[i])
            axes[i].set_title(titles_list[i], fontsize=8)
        axes[i].axis('off') # Esconde os eixos mesmo nas células vazias
        
    plt.tight_layout()
    plt.show()

def load_ground_truth(labels_path):
    """
    Lê o ficheiro labels.txt. Formato: [Nome] [N_Digitos] [L X Y W H] ...
    """
    gt_data = {}
    if not os.path.exists(labels_path):
        return {}
    
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            filename = parts[0]
            boxes = []
            
            # Nas tuas versões A e D, o segundo valor é sempre o número de dígitos
            num_digits = int(parts[1])
            cursor = 2
                
            for _ in range(num_digits):
                try:
                    # L, X, Y, W, H
                    lbl, x, y, w, h = map(int, parts[cursor:cursor+5])
                    boxes.append([x, y, x+w, y+h, lbl]) # Convertemos para [x1, y1, x2, y2, label]
                    cursor += 5
                except: break
            gt_data[filename] = boxes
    return gt_data

def process_version(config, model, device, transform, limit=None):
    print(f"\n>>> A PROCESSAR: {config['name']}")
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    ground_truth = load_ground_truth(config['labels_file'])
    image_files = glob.glob(os.path.join(config['images_dir'], "*.png"))
    if limit: image_files = image_files[:limit]

    total_tp, total_fp, total_gt = 0, 0, 0
    grid_images, grid_titles = [], []

    # Parâmetros de deteção (Podes ajustar estes se a Versão A continuar difícil)
    WINDOW_SIZE = (32, 32)
    STEP_SIZE = 4
    CONF_THRESH = 0.91 
    ENTROPY_THRESH = 0.1
    IOU_THRESH = 0.05
    PIXEL_THRESHOLD = 0.03

    for img_path in tqdm(image_files, desc=f"Scanning {config['name']}", leave=False):
        filename = os.path.basename(img_path)
        all_boxes, all_scores, all_labels = [], [], []
        
        img = Image.open(img_path).convert('L')
        draw_img = img.convert('RGB')
        draw = ImageDraw.Draw(draw_img)

        for (x, y, window) in sliding_window(img, STEP_SIZE, WINDOW_SIZE):
            if transforms.ToTensor()(window).mean() < PIXEL_THRESHOLD:
                continue

            input_tensor = transform(window).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                entropy = calculate_entropy(probs)
            
            if conf.item() >= CONF_THRESH and entropy < ENTROPY_THRESH:
                all_boxes.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]])
                all_scores.append(conf.item())
                all_labels.append(pred.item())

        final_preds = []
        if all_boxes:
            keep = nms(torch.tensor(all_boxes, dtype=torch.float32), 
                       torch.tensor(all_scores, dtype=torch.float32), IOU_THRESH)
            for idx in keep:
                idx = idx.item()
                final_preds.append((all_boxes[idx], all_labels[idx]))
                draw.rectangle(all_boxes[idx], outline="red", width=1)
                draw.text((all_boxes[idx][0], all_boxes[idx][1]-10), str(all_labels[idx]), fill="red")

        # Avaliação
        if filename in ground_truth:
            gt_boxes = ground_truth[filename]
            total_gt += len(gt_boxes)
            matched_gt = [False] * len(gt_boxes)
            for p_box, p_lbl in final_preds:
                found = False
                for i, g_box in enumerate(gt_boxes):
                    # IOU > 0.3 é suficiente para considerar que encontrou o dígito
                    if p_lbl == g_box[4] and calculate_iou(p_box, g_box[:4]) > 0.3:
                        if not matched_gt[i]:
                            matched_gt[i] = True
                            total_tp += 1
                            found = True
                            break
                if not found: total_fp += 1

        draw_img.save(os.path.join(output_dir, f"res_{filename}"))
        if len(grid_images) < 9:
            grid_images.append(np.array(draw_img))
            grid_titles.append(f"{filename}\nDet: {len(final_preds)}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "gt": total_gt, "imgs": grid_images, "titles": grid_titles}

def main():
    CHECKPOINT_PATH = './experiments/checkpoint.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(CHECKPOINT_PATH, device)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    LIMITAR_TESTE = None # Definir para um número para limitar o número de imagens testadas

    tasks = [
        {"name": "Versão A", "images_dir": "./Dataset_Cenas_Versão_A/test/images/", "labels_file": "./Dataset_Cenas_Versão_A/test/labels.txt", "output_dir": "./results_sliding_window/Versao_A"},
        {"name": "Versão D", "images_dir": "./Dataset_Cenas_Versão_D/test/images/", "labels_file": "./Dataset_Cenas_Versão_D/test/labels.txt", "output_dir": "./results_sliding_window/Versao_D"}
    ]

    results_summary = []
    for task in tasks:
        res = process_version(task, model, device, transform, limit=LIMITAR_TESTE)
        results_summary.append((task['name'], res))

    # --- RELATÓRIO FINAL ---
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
        visualize_grid(res['imgs'], res['titles'])

if __name__ == '__main__':
    main()