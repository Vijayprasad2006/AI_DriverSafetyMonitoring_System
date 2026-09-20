# DriveGuard AI — Model Cards & Technical Specifications

## 1. System Overview

DriveGuard AI employs a hybrid edge computer vision pipeline combining geometric landmark analysis with deep convolutional neural networks (CNNs) for multi-signal driver vigilance monitoring.

---

## 2. Model Cards

### Model 1: MediaPipe Face Mesh & Landmarker
- **Model Type:** Single-shot detector and 3D dense surface landmark regressor.
- **Backbone:** BlazeFace detector paired with a lightweight MobileNet-based landmark network.
- **Input:** 640×480 BGR/RGB webcam video frames.
- **Output:** 468 canonical 3D coordinates (with optional 10 iris refinement points).
- **Inference Speed:** ~12–18 ms on modern multi-core x86 CPU.
- **Use in DriveGuard:**
  - Extracts 6-point eyelid contours for Eye Aspect Ratio (EAR).
  - Extracts inner/outer lip contours for Mouth Aspect Ratio (MAR).
  - Supplies 6 canonical 3D points for `solvePnP` head pose estimation.

### Model 2: EyeStateCNN (Eye State Classifier)
- **Model Type:** Custom 3-block 2D Convolutional Neural Network with Batch Normalization and Dropout.
- **Input:** 32×32 Grayscale eye patch normalized with mean 0.5, std 0.5.
- **Output:** Binary classification probabilities: `[P(Closed), P(Open)]`.
- **Architecture Details:**
  - Conv2D(1, 16, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
  - Conv2D(16, 32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
  - Conv2D(32, 64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
  - Linear(1024, 64) -> Dropout(0.3) -> Linear(64, 2)
- **Parameters:** ~78,000 trainable weights (~310 KB).
- **Inference Speed:** ~1.1 ms on CPU.
- **Primary Metric:** Balanced Accuracy and F1-Score on MRL Eye Dataset / CEW test splits.

### Model 3: EmotionCNN (Facial Expression Recognition)
- **Model Type:** MobileNetV3-Small transfer learning backbone fine-tuned for 7 facial expression classes.
- **Input:** 224×224 RGB cropped face image normalized with ImageNet statistics.
- **Output:** 7-way probability distribution: `Neutral`, `Happy`, `Sad`, `Surprised`, `Angry`, `Fearful`, `Disgusted`.
- **Parameters:** ~1.8 million parameters (~9.8 MB).
- **Inference Speed:** ~10–14 ms on CPU.
- **Primary Metric:** Macro F1-Score on FER-2013 / AffectNet.

### Model 4: Head Pose Estimator (Geometric PnP)
- **Method:** Iterative Levenberg-Marquardt `cv2.solvePnP` fitting a 3D anthropometric facial model to 2D image coordinates.
- **Output:** Tait-Bryan Euler angles: Pitch (nodding/slump), Yaw (lateral distraction), Roll (side tilt).
- **Latency:** <0.4 ms per frame.

---

## 3. Ethical Considerations & Disclaimers

> [!IMPORTANT]
> **Research Prototype Notice:** DriveGuard AI is designed for driver vigilance research and simulation. It is **not an automotive-certified safety actuator** (such as ISO 26262 ASIL). It must not be deployed to actuate physical brakes or steering on public roads.

> [!NOTE]
> **Non-Diagnostic Facial Expression Classification:** Emotion predictions represent outward visual muscle configurations classified by machine learning models. They do not constitute psychological diagnoses, mental health evaluations, or definitive proofs of driver anger or sadness. Emotion predictions are displayed contextually and **never independently trigger critical safety intervention**.

---

## 4. Operational Limitations

1. **Occlusion & Eyewear:** Heavy dark sunglasses or reflective prescription lenses may degrade optical EAR calculation. The system mitigates this by fusing landmark geometry with the deep learning eye classifier patch.
2. **Extreme Head Poses:** When head yaw exceeds 45°, face landmarks may experience partial self-occlusion. The risk engine flags severe yaw as distraction.
3. **Low-Light / Night Driving:** Performance depends on webcam sensor sensitivity and cabin lighting. Near-infrared (NIR) illuminators are recommended for production night driving environments.
