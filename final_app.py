import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageFont
import os

# --- CONFIGURARE ---
MODEL_PATH = "models2/best-int8.tflite"
LABEL_PATH = "models2/classes.txt"

INPUT_DIR = "dataset_mare"       
OUTPUT_DIR = "rezultate_totale"  

# --- PARAMETRI ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def dequantize(value, scale, zero_point):
    if scale > 0:
        return (float(value) - zero_point) * scale
    return float(value)

def load_dynamic_font(img_height):
    target_size = max(15, int(img_height / 30))
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, target_size)
            except: continue
    return ImageFont.load_default()

def main():
    print(f"--- Start Procesare (Threshold: {CONF_THRESHOLD}) ---")
    
    # Incarcare Model
    try:
        ext_delegate = tflite.load_delegate("/usr/lib/libvx_delegate.so")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[ext_delegate])
        print("Run on NPU")
    except:
        print("WARNING: No NPU found. Slower CPU run.")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    out_scale, out_zero_point = output_details[0]['quantization']
    
    # Incarcare Etichete
    labels = [line.strip() for line in open(LABEL_PATH)]

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Cautare Imagini
    images_found = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.')]
    print(f"Starting to process {len(images_found)} images")

    count = 0
    for filename in images_found:
        try:
            img_path = os.path.join(INPUT_DIR, filename)
            img_orig = Image.open(img_path).convert('RGB')
            
            font = load_dynamic_font(img_orig.height)
            line_thickness = max(2, int(img_orig.width / 200))

            img_resized = img_orig.resize((320, 320))
            input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            boxes, confidences, class_ids = [], [], []
            for i in range(output_data.shape[0]):
                row = output_data[i]
                conf = dequantize(row[4], out_scale, out_zero_point)
                
                if conf > CONF_THRESHOLD:
                    classes_scores = row[5:]
                    cls_id = np.argmax(classes_scores)
                    cls_score = dequantize(classes_scores[cls_id], out_scale, out_zero_point)
                    score = conf * cls_score
                    
                    if score > CONF_THRESHOLD:
                        x, y, w, h = [dequantize(row[j], out_scale, out_zero_point) for j in range(4)]
                        boxes.append([int((x-w/2)*320), int((y-h/2)*320), int(w*320), int(h*320)])
                        confidences.append(float(score))
                        class_ids.append(cls_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
            draw = ImageDraw.Draw(img_orig)
            
            faces_in_img = 0
            if len(indices) > 0:
                for i in indices.flatten():
                    faces_in_img += 1
                    box = boxes[i]
                    x_scale, y_scale = img_orig.width/320, img_orig.height/320
                    l, t = int(box[0]*x_scale), int(box[1]*y_scale)
                    r, b = int((box[0]+box[2])*x_scale), int((box[1]+box[3])*y_scale)
                    
                    lbl = labels[class_ids[i]]
                    col = "#00FF00" if "helmet" in lbl.lower() else "#FF0000"
                    
                    draw.rectangle([l, t, r, b], outline=col, width=line_thickness)
                    
                    txt = f"{lbl.upper()} {confidences[i]:.0%}"
                    try:
                        tb = draw.textbbox((l, t), txt, font=font)
                        draw.rectangle([l, t-(tb[3]-tb[1])-5, l+(tb[2]-tb[0])+10, t], fill=col)
                        draw.text((l+5, t-(tb[3]-tb[1])-5), txt, fill="white", font=font)
                    except:
                        pass # Fallback pentru versiuni vechi de Pillow

            save_path = os.path.join(OUTPUT_DIR, "rez_" + filename)
            img_orig.save(save_path)
            
            count += 1
            if count % 10 == 0:
                print(f"Procesat {count}/{len(images_found)} imagini...")
            
        except Exception as e:
            print(f"Eroare la {filename}: {e}")


if __name__ == "__main__":
    main()