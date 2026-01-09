# EdgeGuard – Real-Time Industrial Safety Monitoring

This repository contains the implementation of an Edge AI system for detecting safety helmet usage in industrial environments, designed to run locally on the NXP i.MX 8M Plus platform using NPU acceleration.

---

## Project Documentation

The attached PDF file contains the complete project documentation, including the problem definition, system motivation, hardware context, and academic references.

Document:
EdgeGuard_Real-Time_Industrial_Safety_Monitoring_using_NPU_Acceleration_on_NXP_i.MX_8M_Plus.pdf

---

## Overview of the Final Implementation

The project was developed in two main stages. This repository focuses on the first stage, which covers dataset preparation, model training, and optimization for edge deployment.

The goal of this stage was to design and train the AI model that will later be deployed on embedded hardware.

---

## 1. Problem Definition and Dataset

The project addresses the problem of monitoring Personal Protective Equipment (PPE) compliance in industrial environments.

Objective:
Train an object detection model capable of distinguishing between:
- head (no safety helmet)
- helmet / hard hat

A custom dataset was prepared and split into:
- Train set for model learning
- Validation set for performance monitoring
- Test set for inference and final evaluation

This phase represents the data engineering component of the project.

---

## 2. Model Training (YOLOv5)

The model was trained using transfer learning based on the YOLOv5 architecture, selected for its balance between accuracy and inference speed.

Training details:
- Framework: PyTorch (YOLOv5)
- Environment: Google Colab with GPU acceleration
- Image size: 320 × 320
- Number of classes: 2
- Training duration: ~100 epochs

Result:
The trained model achieved a mAP@50 score of approximately 87.4%, confirming that it learned to reliably detect helmet usage.

---

## 3. Optimization for Embedded Deployment

The trained PyTorch model (`best.pt`) was not suitable for direct deployment on the target hardware due to its size and floating-point representation.

Hardware constraints:
- Target board: NXP i.MX 8M Plus
- Accelerator: VeriSilicon Neural Processing Unit (NPU)
- Preferred data type: INT8

To address this, post-training quantization was applied.

Optimization steps:
- Conversion from PyTorch to TensorFlow Lite
- Quantization of weights and activations to INT8
- Generation of an NPU-compatible `.tflite` model

This process significantly reduced model size and enabled efficient execution on the NPU with minimal accuracy loss.

---

## Conceptual Pipeline

Camera input  
→ Image preprocessing  
→ Quantized YOLOv5 model (TensorFlow Lite, INT8)  
→ VeriSilicon NPU  
→ Real-time helmet detection

---

## Conclusion

In this stage of the project:
- The AI model was designed and trained in a cloud environment
- Model performance was validated using standard object detection metrics
- The model was optimized for deployment on embedded hardware

The next stage focuses on deploying the quantized model on the NXP i.MX 8M Plus board and integrating real-time inference and alert logic.

---

## Team

- Alexandra Bucătaru – Dataset preparation and model training
- Robert Rășcanu – Embedded deployment and hardware integration
