# DriveGuard AI — Dataset Acquisition & Training Guide

This guide describes how to acquire benchmark datasets, structure directories, and run training and evaluation workflows for the Eye State and Facial Emotion models.

---

## 1. Supported Datasets

### A. MRL Eye Dataset
- **Source:** [Media Research Lab (MRL) Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) / [Kaggle MRL Eye](https://www.kaggle.com/datasets/taufikmuhammad/mrl-eye-dataset)
- **Content:** Over 84,000 eye images capturing infrared and visible conditions, reflections, and glasses across diverse subjects.
- **Classes:** `Closed` (0) and `Open` (1).

### B. CEW (Closed Eyes in the Wild)
- **Source:** [CEW Dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html)
- **Content:** 4,846 eye images cropped from real-world photos.

### C. FER-2013 (Facial Expression Recognition)
- **Source:** [Kaggle Challenges in Machine Learning: Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **Content:** 35,887 48×48 grayscale/RGB face crops categorized into 7 emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

---

## 2. Directory Structure

Place raw or preprocessed datasets under the `data/` directory:

```
DriveGuard-AI/
└── data/
    ├── eye_dataset/
    │   ├── Closed/
    │   │   ├── closed_0001.png
    │   │   └── ...
    │   └── Open/
    │       ├── open_0001.png
    │       └── ...
    │
    └── fer_dataset/
        ├── Neutral/
        ├── Happy/
        ├── Sad/
        ├── Surprised/
        ├── Angry/
        ├── Fearful/
        └── Disgusted/
```

> [!NOTE]
> If these directories are absent or empty when running the training scripts, the scripts will automatically generate reproducible synthetic samples to demonstrate the end-to-end training and checkpointing pipeline without blocking execution.

---

## 3. Training Commands

### Training the Eye State Classifier
```powershell
python training/train_eye_model.py --dataset_dir data/eye_dataset --epochs 10 --batch_size 32 --lr 0.001 --device cpu
```
Checkpoints will be saved to `models/weights/eye_model.pth`.

### Fine-Tuning the Facial Emotion Classifier
```powershell
python training/train_emotion_model.py --dataset_dir data/fer_dataset --epochs 5 --batch_size 16 --lr 0.0003 --device cpu
```
Checkpoints will be saved to `models/weights/emotion_model.pth`.

---

## 4. Model Evaluation & Benchmarking

To evaluate trained weights against a holdout test split and generate an empirical report for the Streamlit dashboard:

```powershell
python training/evaluate_models.py --dataset_dir data/eye_dataset --weights_path models/weights/eye_model.pth
```

This updates `database/evaluation_report.json` with genuine Accuracy, Precision, Recall, F1-Score, and Confusion Matrix metrics.
