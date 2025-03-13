# **Artificial Intelligence-Based Screening for Blood Pressure Phenotypes of White-Coat and Masked Hypertension in Outpatient Settings**
**MING HUI HUNG, MD, Chun-Hung Chen, MD, Yu-Hsuan Tseng, MD, and Chin-Chou Huang, MD, PhD**  
Published in *Circulation*, 150 (Abstract) A4145524-A4145524 (2024).  

> **Status:** Currently under submission for the original paper.

---

## **Introduction**
White-coat hypertension (WCH) and masked hypertension (MH) complicate accurate blood pressure (BP) monitoring. While **ambulatory BP monitoring (ABPM)** is effective, its high cost and limited availability pose barriers.  

### **Hypothesis**
We hypothesized that a **machine learning (ML) model** using clinical data from a **single outpatient visit** could accurately predict WCH and MH.

---

## **Aims**
This study aimed to **develop and validate ML-based prediction models** for WCH and MH using accessible clinical data, improving **diagnostic efficiency and accessibility**.

---

## **Methods**
### **Data Collection**
- Patients were enrolled from two hypertension cohorts.
- Exclusion criteria: **Incomplete data**.
- Classification: Based on **office BP** and **ABPM readings**, following **American Heart Association (AHA) guidelines**.

### **Machine Learning Models**
The following **ML models** were developed and evaluated:
- **Multi-layer Perceptron (MLP)**
- **Support Vector Machine (SVM)**
- **Tabular Prior-Data Fitted Network (TabPFN)**  

#### **Features Used**
- **Demographics**: Age, gender, height, weight, smoking status.
- **Clinical data**: Office blood pressure (OBP) and heart rate (HR).
- **Feature engineering**: Applied **Principal Component Analysis (PCA)**, **kernel PCA (kPCA)**, and **t-distributed stochastic neighbor embedding (t-SNE)** to improve class separability.

---

## **Results**
### **Study Population**
- **Total participants**: 1,481
- **Mean age**: 47.6 years (**SD**: 13.6)
- **Gender distribution**: 65% male
- **Smokers**: 20.1%

### **BP Measurements**
| Measurement | Mean ± SD |
|------------|----------|
| **Office SBP** | 128.7 ± 15.4 mmHg |
| **Office DBP** | 84.2 ± 11.6 mmHg |
| **24-hour ABPM SBP** | 122.5 ± 11.8 mmHg |
| **24-hour ABPM DBP** | 79.3 ± 10.1 mmHg |

### **Model Performance**
The **TabPFN model** achieved the best predictive performance:  

#### **White-Coat Hypertension (WCH)**
- **Recall**: 0.747  
- **Precision**: 0.931  
- **F1 Score**: 0.829  
- **Accuracy**: 0.807  

#### **Masked Hypertension (MH)**
- **Recall**: 0.713  
- **Precision**: 0.954  
- **F1 Score**: 0.816  
- **Accuracy**: 0.907  

### **ROC Curves**
#### **White-Coat Hypertension (WCH)**
![ROC Curve for WCH](roc_curve_WCH_filter.png)

#### **Masked Hypertension (MH)**
![ROC Curve for MH](roc_curve_MH_filter.png)

---

## **Conclusion**
Our **ML-based prediction model** effectively identifies WCH and MH using **readily accessible clinical data**.  
This approach provides a **cost-effective alternative** before **ABPM**, helping clinicians make better-informed decisions.

---
