import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Paths for model, labels, test image and NPU delegate
MODEL_PATH = "models/detect.tflite"
LABEL_PATH = "models/labelmap.txt"
IMAGE_PATH = "test_image.jpg"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

def main():
    print("Optimization test started: SSD MobileNet V2")

    # Load TFLite model with NPU delegate
    print(f"Loading delegate: {DELEGATE_PATH}")
    try:
        delegate = tflite.load_delegate(DELEGATE_PATH)
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
    except Exception as e:
        print(f"Failed to initialize NPU: {e}")
        return

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    print(f"Model input resolution: {width}x{height}")

    # Load and resize input image
    image = Image.open(IMAGE_PATH).convert("RGB")
    resized = image.resize((width, height))
    input_data = np.expand_dims(resized, axis=0)

    # Warm-up runs to stabilize NPU performance
    interpreter.set_tensor(input_details[0]["index"], input_data)
    print("Warming up NPU (3 runs)")
    for _ in range(3):
        interpreter.invoke()

    # Benchmark loop
    print("Running benchmark (50 iterations)")
    iterations = 50
    start_time = time.time()

    for _ in range(iterations):
        interpreter.invoke()

    total_time = time.time() - start_time
    avg_ms = (total_time / iterations) * 1000
    fps = 1.0 / (total_time / iterations)

    # SSD MobileNet output format:
    # 0 = bounding boxes, 1 = classes, 2 = scores, 3 = number of detections
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    print("\n" + "=" * 30)
    print("RESULTS")
    print(f"Average inference time: {avg_ms:.2f} ms")
    print(f"Estimated FPS: {fps:.2f}")
    print("=" * 30)

    # Load label map
    labels = {}
    with open(LABEL_PATH, "r") as f:
        for idx, line in enumerate(f):
            labels[idx] = line.strip()

    print("Detections above 50% confidence:")
    for i in range(len(scores)):
        if scores[i] > 0.5:
            class_id = int(classes[i])
            label = labels.get(class_id, "Object")
            print(f" -> {label}: {scores[i] * 100:.1f}%")

if __name__ == "__main__":
    main()
