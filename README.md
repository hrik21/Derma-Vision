# 🩺 DermaVision: Skin Lesion Classification Using Deep Learning

## 📘 Overview
**DermaVision** is a deep learning–based project focused on classifying **skin lesions** from dermoscopic images to aid in early detection of **melanoma** and other skin cancers.  
The project explores and compares three convolutional neural network architectures—**Baseline CNN**, **Custom CNN**, and **ResNet50 (transfer learning)**—using the **HAM10000** dataset.

This research contributes to improving diagnostic precision and automation in dermatology, where accurate lesion classification can significantly impact patient outcomes.

---

## 🎯 Objectives
- Develop a **robust CNN-based classifier** for multi-class skin lesion classification.  
- Leverage **transfer learning** with **ResNet50** for better generalization.  
- Compare different architectures on metrics such as **accuracy, precision, recall, F1-score, and loss**.  
- Advance **AI-driven diagnostic tools** to assist dermatologists in early skin cancer detection.

---

## 🧠 Dataset
**Dataset:** [HAM10000 (Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)  
**Description:** 10,015 dermoscopic images of pigmented skin lesions across **7 diagnostic categories**.

| Split | No. of Images | Description |
|:------|:--------------|:-------------|
| Training | 5,187 | Model training |
| Validation | 1,556 | Hyperparameter tuning |
| Testing | 667 | Final evaluation |

All images were resized to **176×176 pixels**, normalized to `[0,1]`, and augmented to improve generalization.

---

## ⚙️ Data Preprocessing
- **Rescaling:** Images resized to a consistent 176×176×3 dimension.  
- **Normalization:** Pixel values scaled between `[0,1]`.  
- **Augmentation:** Applied random rotations, flips, and shifts to mitigate overfitting.  
- **Stratified Splitting:** Ensured class balance across train, validation, and test sets.

---

## 🧩 Model Architectures

### 🧱 **Model 1 — Baseline CNN**
A simple CNN with:
- 2 Convolutional layers
- MaxPooling, Dropout, and Dense layers
- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Peak Training Accuracy:** 85.23%  
- **Validation Accuracy:** 69.53%  
- **Test Accuracy:** 66.26%

---

### 🧱 **Model 2 — Custom CNN**
Enhanced architecture with:
- 4 Convolutional layers (32–256 filters)
- MaxPooling and Dropout for regularization
- Dense layer of 1024 neurons  
- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Peak Training Accuracy:** 73.85%  
- **Validation Accuracy:** 68.05%  
- **Test Accuracy:** 67.76%

---

### 🧱 **Model 3 — ResNet50 (Transfer Learning)**
A fine-tuned **ResNet50** network pre-trained on ImageNet.
- Input Shape: (176,176,3)
- 23.6M parameters (23.5M trainable)
- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Training Accuracy:** 60.18%  
- **Validation Accuracy:** 57.51%  
- **Test Accuracy:** 57.87%

---

## 📊 Results Summary

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|:------|:------------------|:--------------------|:--------------|
| Baseline CNN | 85.23% | 69.53% | 66.26% |
| Custom CNN | 73.85% | 68.05% | **67.76%** |
| ResNet50 | 60.18% | 57.51% | 57.87% |

**Best Performing Model:** Custom CNN (Test Accuracy 67.76%)  
The custom CNN balanced complexity and generalization, outperforming both the baseline and transfer learning models.

---

## 📈 Evaluation Metrics
- **Accuracy:** Ratio of correctly predicted classes  
- **Precision, Recall, F1-Score:** Measured per class  
- **Loss Function:** Categorical Cross-Entropy  
- **Visualization:** Accuracy/Loss curves, confusion matrices, and classification reports  

---

## 🔍 Key Findings
- **Custom CNN** achieved the best trade-off between accuracy and computational cost.  
- **ResNet50** underperformed due to overfitting on small dataset size despite using transfer learning.  
- Proper preprocessing and data balancing significantly improved generalization.  
- The system shows promise for **early detection of melanoma**, supporting dermatologists in diagnostic workflows.

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Frameworks:** TensorFlow, Keras, Scikit-learn  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 🚀 Future Work
- Integrate **attention mechanisms** or **Vision Transformers (ViT)** for enhanced feature extraction.  
- Expand to **multi-modal analysis** combining dermoscopic and clinical images.  
- Deploy as a **Streamlit-based web app** for real-time skin lesion classification.  
- Incorporate **Grad-CAM** visualizations to highlight lesion features influencing predictions.

---

## 🧾 References
1. Tschandl, P. *et al.*, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Harvard Dataverse, 2018.  
2. Papers With Code — [HAM10000 Dataset](https://paperswithcode.com/dataset/ham10000)  
3. Activeloop — [HAM10000 Dataset](https://datasets.activeloop.ai/dataset/ham10000)

---

## 👨‍💻 Authors
- **Hrithik Puri** — [LinkedIn](https://www.linkedin.com/in/puri-hrithik/) | [GitHub](https://github.com/hrik21)  
- **Shubham Pandey** — Northeastern University  
- **Atharva Avinash Gosavi** — Northeastern University  

---

## 💡 Conclusion
DermaVision demonstrates how **deep learning can augment dermatological diagnostics** by automating lesion classification.  
While accuracy is constrained by dataset diversity, the project highlights the feasibility of CNN-based systems in clinical decision support and sets a foundation for more advanced AI-driven diagnostic tools.

---
