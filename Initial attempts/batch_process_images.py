import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

# Paths for model, labels, font and NPU delegate
MODEL_PATH = "models/detect.tflite"
LABEL_PATH = "models/labelmap.txt"
FONT_PATH = "models/BoldFont.ttf"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# Input and output folders
INPUT_DIR = "test_images"
OUTPUT_DIR = "results_images"

# Visualization tuning
FONT_SIZE = 18
BOX_THICKNESS = 3

# Load font for drawing labels
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print("Warning: custom font not found, using default font")
    font = ImageFont.load_default()

def load_interpreter():
    try:
        delegate = tflite.load_delegate(DELEGATE_PATH)
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Critical NPU error: {e}")
        exit(1)

def process_single_image(interpreter, image_path, output_path, labels):
    try:
        image = Image.open(image_path).convert("RGB")
    except IOError:
        print(f"Failed to open image: {image_path}")
        return

    input_details = interpreter.get_input_details()
    model_height = input_details[0]["shape"][1]
    model_width = input_details[0]["shape"][2]

    resized = image.resize((model_width, model_height))
    input_data = np.expand_dims(resized, axis=0)

    start_time = time.time()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000

    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    detections = 0
    for i in range(len(scores)):
        if scores[i] > 0.5:
            detections += 1

            ymin, xmin, ymax, xmax = boxes[i]
            left = xmin * img_width
            top = ymin * img_height
            right = xmax * img_width
            bottom = ymax * img_height

            class_id = int(classes[i])
            label = labels.get(class_id, "Object")

            draw.rectangle(
                [left, top, right, bottom],
                outline="lime",
                width=BOX_THICKNESS
            )

            text = f"{label} {scores[i] * 100:.0f}%"
            text_box = draw.textbbox((left, top), text, font=font)

            draw.rectangle(
                [
                    text_box[0] - 1,
                    text_box[1] - 1,
                    text_box[2] + 1,
                    text_box[3] + 1
                ],
                fill="black"
            )

            draw.text(
                (left, top),
                text,
                fill="lime",
                font=font
            )

    image.save(output_path)
    print(
        f"Processed {os.path.basename(image_path)} | "
        f"{inference_time:.1f} ms | "
        f"Detections: {detections}"
    )

def main():
    print("Image processing started")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    interpreter = load_interpreter()

    labels = {}
    with open(LABEL_PATH, "r") as f:
        for idx, line in enumerate(f):
            labels[idx] = line.strip()

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, "result_" + filename)
        process_single_image(interpreter, input_path, output_path, labels)

    print(f"Done. Results available in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
