# Brain Tumor Classification Using Deep Learning

## Overview
This project presents a deep learning-based brain tumor classification system using MRI images and transfer learning with the VGG16 architecture. The model classifies brain MRI scans into four categories and is deployed using Streamlit for interactive predictions.

## Features
- Multi-class brain tumor classification
- MRI image-based prediction system
- Transfer learning using VGG16
- Streamlit web application
- Real-time prediction interface
- Performance evaluation using accuracy, precision, recall, and F1-score

## Tumor Classes
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Model Architecture
- VGG16 Transfer Learning
- Fine-tuned convolutional layers
- Global Average Pooling
- Dropout Regularization
- Batch Normalization
- Softmax Classification

## Performance
- Test Accuracy: 98.4%
- High precision and recall across all classes
- Strong generalization on MRI image dataset

## Dataset
The model was trained on a publicly available Brain Tumor MRI dataset containing MRI images categorized into:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL

## Project Structure

```text
Brain-Tumor-Classification/
│
├── app.py
├── Brain_Tumor_Classification.ipynb
├── requirements.txt
├── README.md
├── LICENSE
├── AIML_Project_Report_PPT.pdf
└── Research_Paper.pdf
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Smruti2205/Brain-Tumor-Classification.git
```

### 2. Navigate to the project folder

```bash
cd Brain-Tumor-Classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Future Improvements
- Integration with advanced CNN architectures
- Multi-sequence MRI analysis
- Tumor segmentation
- Clinical deployment support
- Explainable AI visualizations

---

## License
This project is licensed under the MIT License.
