import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

# Paths to model, labels, font and NPU delegate
MODEL_PATH = "models/detect.tflite"
LABEL_PATH = "models/labelmap.txt"
FONT_PATH = "models/BoldFont.ttf"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# Input and output video files
VIDEO_INPUT = "test_video.mp4"
VIDEO_OUTPUT = "result_video.avi"

# Drawing settings
FONT_SIZE = 18
BOX_THICKNESS = 3

# Load font used for labels
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    font = ImageFont.load_default()

def main():
    print("Starting video processing")

    # Load TFLite model using the NPU delegate
    delegate = tflite.load_delegate(DELEGATE_PATH)
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_height = input_details[0]["shape"][1]
    model_width = input_details[0]["shape"][2]

    # Load class labels
    labels = {}
    with open(LABEL_PATH, "r") as f:
        for idx, line in enumerate(f):
            labels[idx] = line.strip()

    # Open input video
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("Failed to open input video")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Input video: {frame_width}x{frame_height} @ {fps} FPS")

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        VIDEO_OUTPUT,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        resized_image = pil_image.resize((model_width, model_height))
        input_tensor = np.expand_dims(resized_image, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        draw = ImageDraw.Draw(pil_image)

        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]

                left = xmin * frame_width
                top = ymin * frame_height
                right = xmax * frame_width
                bottom = ymax * frame_height

                class_id = int(classes[i])
                label = labels.get(class_id, "Object")
                confidence = scores[i] * 100

                draw.rectangle(
                    [left, top, right, bottom],
                    outline="lime",
                    width=BOX_THICKNESS
                )

                text = f"{label} {confidence:.0f}%"
                text_box = draw.textbbox((left, top), text, font=font)

                draw.rectangle(text_box, fill="black")
                draw.text((left, top), text, fill="lime", font=font)

        output_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        out.write(output_frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"Finished processing {frame_count} frames in {elapsed:.2f}s")
    print(f"Output saved as '{VIDEO_OUTPUT}'")

if __name__ == "__main__":
    main()
