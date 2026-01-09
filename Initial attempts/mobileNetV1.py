import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Paths for model, labels, test image and NPU delegate
MODEL_PATH = "models/mobilenet_v1_1.0_224_quant.tflite"
LABEL_PATH = "models/labels_mobilenet_quant_v1_224.txt"
IMAGE_PATH = "test_image.jpg"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def main():
    print("NXP i.MX 8M Plus speed test")

    # Load model with NPU delegate if available
    try:
        delegate = tflite.load_delegate(DELEGATE_PATH)
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[delegate]
        )
        print("NPU delegate loaded successfully")
    except Exception:
        print("NPU not available, falling back to CPU")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    image = Image.open(IMAGE_PATH).convert("RGB")
    resized = image.resize((width, height))
    input_data = np.expand_dims(resized, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Initial run
    interpreter.invoke()

    start_time = time.time()

    for _ in range(10):
        interpreter.invoke()

    avg_time_ms = (time.time() - start_time) / 10 * 1000

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    labels = load_labels(LABEL_PATH)

    top_index = np.argmax(output_data)

    print("\nFINAL RESULT")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print("-" * 30)
    print(f"Detected class: {labels[top_index]}")
    print(f"Confidence: {(output_data[top_index] / 255.0) * 100:.2f}%")

if __name__ == "__main__":
    main()
