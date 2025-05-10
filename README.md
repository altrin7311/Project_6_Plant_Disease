# 🌿 Plant Disease Classification with EfficientNet & DenseNet



https://github.com/user-attachments/assets/6f860eb4-3e7f-45e8-b1b2-376d90f3e0ca



This project leverages deep learning to identify plant diseases from leaf images. Using transfer learning with EfficientNetB0 and DenseNet121, the model can classify images into 8 specific categories. A Streamlit-based web interface allows users to upload images and get instant predictions.

---

## 📁 Dataset Structure

The dataset is organized into the following folders:

```
Project_6_Plant_Disease/
├── train/     # Training images (8 classes)
├── valid/     # Validation images
```

Each folder contains subdirectories for the 8 selected classes.

---

## ✅ Disease Categories

1. Apple___Cedar_apple_rust  
2. Corn_(maize)___Northern_Leaf_Blight  
3. Grape___Black_rot  
4. Potato___Late_blight  
5. Tomato___Early_blight  
6. Tomato___Target_Spot  
7. Tomato___Tomato_Yellow_Leaf_Curl_Virus  
8. Tomato___healthy  

---

## 🧠 Models Used

- **EfficientNetB0** (model_efficientnet.keras)
- **DenseNet121** (model_densenet.keras)

These models were trained with an input size of 128x128 and evaluated using classification reports and confusion matrices.

---

## 🔧 Setup Instructions

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # or use venv\Scripts\activate on Windows
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python train_model_efficientnet.py
python train_model_densenet.py
```

### 4. Evaluate the models

```bash
python evaluation_summary.py
```

### 5. Launch the web app

```bash
streamlit run app/app.py
```

---

## 📊 Evaluation Results

EfficientNetB0 achieved a validation accuracy of up to **98%**. Evaluation includes:

- Confusion matrix visualizations
- Detailed classification reports (precision, recall, F1-score)

---

## 📦 Project Structure

```
Project_6_Plant_Disease/
│
├── app/                      # Streamlit app UI
├── models/                   # Trained models (.keras)
├── train/                    # Training dataset
├── valid/                    # Validation dataset
├── test/                     # Upload folder for predictions
├── train_model_efficientnet.py
├── train_model_densenet.py
├── evaluation_summary.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## 📌 Features

- EfficientNet & DenseNet-based classifiers
- Preprocessing with `image_dataset_from_directory`
- Real-time prediction via Streamlit
- Confusion matrix and metric visualization

---

## 👨‍🔬 Author

Built by Altrin Titus for plant disease classification using TensorFlow and Streamlit.
