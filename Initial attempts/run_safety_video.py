import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

# Model, labels, font and NPU delegate paths
MODEL_PATH = "models/safety.tflite"
LABEL_PATH = "models/safety_labels.txt"
FONT_PATH = "models/BoldFont.ttf"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# Input and output video
VIDEO_INPUT = "test_video.mp4"
VIDEO_OUTPUT = "final_demo.avi"

# Detection thresholds
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

# Visualization settings
FONT_SIZE = 18
BOX_THICKNESS = 3
COLOR_HEAD = (255, 0, 0)
COLOR_HELMET = (0, 255, 0)

# Load font for drawing labels
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
    print("Safety monitoring started")

    # Load TFLite model with NPU delegate
    try:
        delegate = tflite.load_delegate(DELEGATE_PATH)
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
    except Exception as e:
        print(f"NPU initialization error: {e}")
        return

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load class labels
    labels = [line.strip() for line in open(LABEL_PATH)]
    print(f"Loaded classes: {labels}")

    # Open input video
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("Failed to open input video")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        VIDEO_OUTPUT,
        fourcc,
        fps,
        (video_width, video_height)
    )

    frame_count = 0
    print("Processing started")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (320, 320))

        if input_details[0]["dtype"] == np.float32:
            input_data = resized.astype(np.float32) / 255.0
        else:
            input_data = resized.astype(np.uint8)

        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        indices, class_ids, confidences, boxes = class_filter(output_data)

        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        if len(indices) > 0:
            for i in indices.flatten():
                left_m, top_m, w_m, h_m = boxes[i]

                x_scale = video_width / 320
                y_scale = video_height / 320

                left = int(left_m * x_scale)
                top = int(top_m * y_scale)
                right = int((left_m + w_m) * x_scale)
                bottom = int((top_m + h_m) * y_scale)

                left = max(0, left)
                top = max(0, top)
                right = min(video_width, right)
                bottom = min(video_height, bottom)

                if left >= right or top >= bottom:
                    continue

                class_name = labels[class_ids[i]].lower()

                has_helmet = "helmet" in class_name and "no" not in class_name

                if has_helmet:
                    color = COLOR_HELMET
                    text = f"HELMET OK ({confidences[i]:.0%})"
                else:
                    color = COLOR_HEAD
                    text = f"NO HELMET ({confidences[i]:.0%})"

                draw.rectangle(
                    [left, top, right, bottom],
                    outline=color,
                    width=BOX_THICKNESS
                )

                text_box = draw.textbbox((left, top), text, font=font)
                draw.rectangle(text_box, fill=color)
                draw.text((left, top), text, fill="black", font=font)

        output_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        out.write(output_frame)

        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print(f"Finished. Output saved as {VIDEO_OUTPUT}")

if __name__ == "__main__":
    main()
