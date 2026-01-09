import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

# Paths for model, labels, font and NPU delegate
MODEL_PATH = "models/safety.tflite"
LABEL_PATH = "models/safety_labels.txt"
FONT_PATH = "models/BoldFont.ttf"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# Input and output folders
INPUT_DIR = "test_images_final"
OUTPUT_DIR = "results_final"

# Detection thresholds
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

# Drawing parameters
FONT_SIZE = 20
BOX_THICKNESS = 4
COLOR_HEAD = (255, 0, 0)
COLOR_HELMET = (0, 255, 0)

# Load font used for labels
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    font = ImageFont.load_default()

def class_filter(output_data):
    classes = []
    confidences = []
    boxes = []

    for i in range(output_data.shape[0]):
        row = output_data[i]
        confidence = row[4]

        if confidence > CONF_THRESHOLD:
            class_scores = row[5:]
            class_id = np.argmax(class_scores)

            if class_scores[class_id] > CONF_THRESHOLD:
                confidences.append(confidence)
                classes.append(class_id)

                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - w / 2) * 320)
                top = int((y - h / 2) * 320)
                width = int(w * 320)
                height = int(h * 320)

                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    return indices, classes, confidences, boxes

def main():
    print("Batch image processing started")

    # Load TFLite model with NPU delegate
    try:
        delegate = tflite.load_delegate(DELEGATE_PATH)
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"NPU initialization failed: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load class labels
    labels = [line.strip() for line in open(LABEL_PATH)]
    print(f"Loaded classes: {labels}")

    # Collect input images
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(files)} images in '{INPUT_DIR}'")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, "result_" + filename)

        try:
            image = Image.open(input_path).convert("RGB")
            orig_width, orig_height = image.size
        except IOError:
            print(f"Failed to open image: {filename}")
            continue

        resized = image.resize((320, 320), Image.Resampling.LANCZOS)

        if input_details[0]["dtype"] == np.float32:
            input_data = np.array(resized, dtype=np.float32) / 255.0
        else:
            input_data = np.array(resized, dtype=np.uint8)

        input_data = np.expand_dims(input_data, axis=0)

        start_time = time.time()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]
        inference_time = (time.time() - start_time) * 1000

        indices, class_ids, confidences, boxes = class_filter(output_data)

        draw = ImageDraw.Draw(image)
        detection_count = 0

        if len(indices) > 0:
            for i in indices.flatten():
                detection_count += 1

                left_m, top_m, w_m, h_m = boxes[i]

                x_scale = orig_width / 320
                y_scale = orig_height / 320

                left = int(left_m * x_scale)
                top = int(top_m * y_scale)
                right = int((left_m + w_m) * x_scale)
                bottom = int((top_m + h_m) * y_scale)

                left = max(0, left)
                top = max(0, top)
                right = min(orig_width, right)
                bottom = min(orig_height, bottom)

                if left >= right or top >= bottom:
                    continue

                class_name = labels[class_ids[i]].lower()

                if "head" in class_name or "no" in class_name:
                    color = COLOR_HEAD
                    text = f"NO HELMET ({confidences[i]:.0%})"
                elif "helmet" in class_name:
                    color = COLOR_HELMET
                    text = f"HELMET OK ({confidences[i]:.0%})"
                else:
                    color = "yellow"
                    text = f"{class_name} ({confidences[i]:.0%})"

                draw.rectangle(
                    [left, top, right, bottom],
                    outline=color,
                    width=BOX_THICKNESS
                )

                text_box = draw.textbbox((left, top), text, font=font)
                draw.rectangle(text_box, fill="black")
                draw.text((left, top), text, fill=color, font=font)

        image.save(output_path)
        print(
            f"Processed {filename} | "
            f"{inference_time:.1f} ms | "
            f"Detections: {detection_count}"
        )

    print(f"Done. Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
