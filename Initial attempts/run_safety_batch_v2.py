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
OUTPUT_DIR = "results_final_v2"

# Detection thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Drawing settings
FONT_SIZE = 18
BOX_THICKNESS = 3
COLOR_HEAD = (255, 0, 0)
COLOR_HELMET = (0, 255, 0)

# Load font used for drawing labels
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    font = ImageFont.load_default()

def dequantize(value, scale, zero_point):
    # Converts an INT8 value back to its real floating-point representation
    if scale > 0:
        return (value - zero_point) * scale
    return float(value)

def class_filter(output_data, scale, zero_point):
    classes = []
    confidences = []
    boxes = []

    num_rows = output_data.shape[0]

    for i in range(num_rows):
        row = output_data[i]

        # Dequantize object confidence
        confidence_raw = row[4]
        confidence = dequantize(confidence_raw, scale, zero_point)

        if confidence > CONF_THRESHOLD:
            class_scores_raw = row[5:]
            class_id = np.argmax(class_scores_raw)

            class_score_raw = class_scores_raw[class_id]
            class_score = dequantize(class_score_raw, scale, zero_point)

            final_score = confidence * class_score

            if final_score > CONF_THRESHOLD:
                confidences.append(float(final_score))
                classes.append(class_id)

                # Dequantize bounding box coordinates
                x = dequantize(row[0], scale, zero_point)
                y = dequantize(row[1], scale, zero_point)
                w = dequantize(row[2], scale, zero_point)
                h = dequantize(row[3], scale, zero_point)

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
    print("Safety check started (INT8 corrected)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Read quantization parameters from model output
    out_scale, out_zero_point = output_details[0]["quantization"]
    print(f"Output quantization: scale={out_scale}, zero_point={out_zero_point}")

    labels = [line.strip() for line in open(LABEL_PATH)]

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {len(files)} images")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, "v2_" + filename)

        try:
            image = Image.open(input_path).convert("RGB")
            orig_width, orig_height = image.size
        except IOError:
            continue

        resized = image.resize((320, 320), Image.Resampling.LANCZOS)

        input_data = np.array(resized, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        indices, class_ids, confidences, boxes = class_filter(
            output_data,
            out_scale,
            out_zero_point
        )

        draw = ImageDraw.Draw(image)
        detections = 0

        if len(indices) > 0:
            for i in indices.flatten():
                detections += 1

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

                class_name = labels[class_ids[i]].lower()

                if "head" in class_name or "no" in class_name:
                    color = COLOR_HEAD
                    text = "NO HELMET"
                elif "helmet" in class_name:
                    color = COLOR_HELMET
                    text = "HELMET OK"
                else:
                    color = "yellow"
                    text = class_name

                text += f" ({confidences[i] * 100:.1f}%)"

                draw.rectangle(
                    [left, top, right, bottom],
                    outline=color,
                    width=BOX_THICKNESS
                )

                text_box = draw.textbbox((left, top), text, font=font)
                draw.rectangle(text_box, fill="black")
                draw.text((left, top), text, fill=color, font=font)

        image.save(output_path)
        print(f"{filename}: {detections} detections")

    print(f"Finished. Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
