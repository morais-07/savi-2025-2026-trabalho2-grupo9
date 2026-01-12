import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import nms 
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from tqdm import tqdm

# Importar o Modelo
from model import ModelBetterCNN 

def load_trained_model(checkpoint_path, device):
    model = ModelBetterCNN().to(device)
    if os.path.exists(checkpoint_path):
        # Adicionado weights_only=False para compatibilidade
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            try: model.load_state_dict(checkpoint)
            except: pass
    else:
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    model.eval()
    return model

def sliding_window(image, step_size, window_size):
    w, h = image.size
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            window = image.crop((x, y, x + window_size[0], y + window_size[1]))
            yield (x, y, window)

def main():
    # ---------------------------------------------------------
    # CONFIGURAÇÕES AJUSTADAS
    # ---------------------------------------------------------
    CHECKPOINT_PATH = './experiments/checkpoint.pkl'
    TEST_IMAGES_DIR = './Dataset_Cenas_Versão_D/test/images/'
    OUTPUT_DIR = './results_sliding_window'
    
    WINDOW_SIZE = (28, 28)
    
    # PASSO PEQUENO (Para não falhar números, gera muitas caixas)
    STEP_SIZE = 12          
    
    # CONFIANÇA MAIS BAIXA (Para apanhar números difíceis)
    CONFIDENCE_THRESHOLD = 0.85 
    
    # VALOR DE INTERSECÇÃO (Se as caixas se tocarem 30%, apaga a pior)
    IOU_THRESHOLD = 0.3    
    
    PIXEL_THRESHOLD = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A usar dispositivo: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_trained_model(CHECKPOINT_PATH, device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))[:5]
    
    print(f"A processar {len(image_files)} imagens...")

    for img_path in image_files:
        original_img = Image.open(img_path).convert('L')
        draw_img = original_img.convert('RGB')
        draw = ImageDraw.Draw(draw_img)
        filename = os.path.basename(img_path)
        
        # Listas para guardar TUDO antes de desenhar
        all_boxes = []
        all_scores = []
        all_labels = []

        # 1. SLIDING WINDOW (Recolha de candidatos)
        total_windows = ((original_img.size[0]-WINDOW_SIZE[0])//STEP_SIZE + 1) * \
                        ((original_img.size[1]-WINDOW_SIZE[1])//STEP_SIZE + 1)

        with tqdm(total=total_windows, desc=f"Scan: {filename}", leave=False) as pbar:
            for (x, y, window) in sliding_window(original_img, STEP_SIZE, WINDOW_SIZE):
                
                # Ignorar preto
                if transforms.ToTensor()(window).sum() < (PIXEL_THRESHOLD / 255.0):
                    pbar.update(1)
                    continue

                input_tensor = transform(window).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, pred_class = torch.max(probs, 1)
                    conf = conf.item()
                    pred_class = pred_class.item()
                
                if conf >= CONFIDENCE_THRESHOLD:
                    # Guardamos: [x1, y1, x2, y2]
                    all_boxes.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]])
                    all_scores.append(conf)
                    all_labels.append(pred_class)
                
                pbar.update(1)

        # 2. APLICAR NMS (A Limpeza)
        if len(all_boxes) > 0:
            # Converter para tensores do PyTorch
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            
            # O NMS devolve os índices das caixas "sobreviventes"
            keep_indices = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)
            
            print(f" -> Encontradas {len(all_boxes)} caixas brutas. Após NMS: {len(keep_indices)}")

            # 3. DESENHAR SÓ AS SOBREVIVENTES
            for idx in keep_indices:
                idx = idx.item() # Converter tensor para int
                box = all_boxes[idx]
                label = all_labels[idx]
                score = all_scores[idx]
                
                # Desenhar
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1-10), f"{label} ({score:.2f})", fill="red")
        else:
            print(" -> Nenhuma deteção encontrada.")

        save_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
        draw_img.save(save_path)

    print("\nConcluído! Verifica a pasta 'results_sliding_window'.")

if __name__ == '__main__':
    main()