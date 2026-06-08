# task-5-mariam-abd-elhady
# Alzheimer's Disease Analysis

## Overview

This project aims to analyze Alzheimer's Disease data and apply machine learning models for predicting the disease. The dataset contains 2,149 samples with 33 features.

## 📊 Dataset Information

### Basic Information
- **Number of samples:** 2,149
- **Number of features:** 33 (after removing PatientID and DoctorInCharge)
- **Target variable:** Diagnosis (0 = Not Diagnosis, 1 = Diagnosis)

### Features Description

| Feature | Description |
|---------|-------------|
| Age | Patient's age |
| Gender | 0: Female, 1: Male |
| Ethnicity | Ethnic group identifier |
| EducationLevel | Level of education |
| BMI | Body Mass Index |
| Smoking | Smoking status (0: No, 1: Yes) |
| AlcoholConsumption | Alcohol consumption level |
| PhysicalActivity | Physical activity level |
| DietQuality | Diet quality score |
| SleepQuality | Sleep quality score |
| FamilyHistoryAlzheimers | Family history of Alzheimer's |
| CardiovascularDisease | Presence of cardiovascular disease |
| Diabetes | Diabetes status |
| Depression | Depression status |
| HeadInjury | Head injury history |
| Hypertension | Hypertension status |
| SystolicBP | Systolic blood pressure |
| DiastolicBP | Diastolic blood pressure |
| CholesterolTotal | Total cholesterol |
| CholesterolLDL | LDL cholesterol |
| CholesterolHDL | HDL cholesterol |
| CholesterolTriglycerides | Triglycerides level |
| MMSE | Mini-Mental State Examination score |
| FunctionalAssessment | Functional assessment score |
| MemoryComplaints | Memory complaints indicator |
| BehavioralProblems | Behavioral problems indicator |
| ADL | Activities of Daily Living |
| Confusion | Confusion indicator |
| Disorientation | Disorientation indicator |
| PersonalityChanges | Personality changes indicator |
| DifficultyCompletingTasks | Difficulty completing tasks indicator |
| Forgetfulness | Forgetfulness indicator |
| Diagnosis | Target variable (0: Not Diagnosis, 1: Diagnosis) |

## 📈 Exploratory Data Analysis (EDA)

### Age Distribution
- Age ranges from approximately 60 to 90+ years
- Shows a relatively normal distribution with a slight right skew
- Most patients are in the 70-85 age range

### BMI vs Diagnosis
- BMI values range from approximately 15 to 45
- No clear linear separation between diagnosed and non-diagnosed patients
- Both groups show overlapping BMI distributions

### Smoking Distribution
- Majority of patients are non-smokers (approximately 75%)
- Smokers represent about 25% of the dataset

### MMSE Score Distribution
- MMSE scores range from 0 to 30
- Shows a bimodal distribution
- Lower scores tend to correlate with Alzheimer's diagnosis

### Physical Activity vs Diagnosis
- Physical activity scores range from 0 to 10
- Higher physical activity appears associated with lower diagnosis rates

### Correlation Matrix
- Strong positive correlations between:
  - MMSE and FunctionalAssessment
  - Various cholesterol measurements
  - Blood pressure measurements (Systolic and Diastolic)
- Negative correlations between:
  - Age and MMSE
  - Age and FunctionalAssessment

### BMI Boxplot
- Shows distribution of BMI values
- Some outliers present on both ends

## 🧹 Data Preprocessing

1. **Removed unnecessary columns:** PatientID, DoctorInCharge
2. **Removed duplicates:** No duplicates found
3. **Handled missing values:** No missing values detected
4. **Outlier removal:** Applied IQR method but no outliers removed from Smoking column
5. **Data splitting:** 80% training, 20% testing
6. **Feature scaling:** Applied StandardScaler to Smoking feature

## 🤖 Machine Learning Models

### Models Implemented

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classification model |
| K-Nearest Neighbors (KNN) | Instance-based learning |
| Decision Tree | Tree-based classification |
| Naive Bayes | Probabilistic classifier |
| Random Forest | Ensemble of decision trees |
| Extra Trees | Extremely randomized trees |
| AdaBoost | Adaptive boosting |
| Gradient Boosting | Gradient boosting algorithm |
| Hist Gradient Boosting | Histogram-based gradient boosting |
| XGBoost | Extreme Gradient Boosting |
| CatBoost | Categorical boosting |
| LightGBM | Light gradient boosting machine |

## 📊 Model Performance Summary

| Model | Test Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|---------------|---------|--------|-----------|--------|----------|
| Logistic Regression | 0.79 | 0.841 | 0.788 | 0.75 | 0.58 | 0.66 |
| KNN | 0.60 | 0.528 | 0.360 | 0.40 | 0.33 | 0.36 |
| Decision Tree | 0.86 | 0.834 | 0.707 | 0.83 | 0.75 | 0.79 |
| Naive Bayes | 0.77 | 0.818 | 0.730 | 0.69 | 0.57 | 0.63 |
| Random Forest | 0.92 | 0.914 | 0.912 | 0.95 | 0.80 | 0.87 |
| Extra Trees | 0.92 | 0.913 | 0.905 | 0.94 | 0.83 | 0.88 |
| AdaBoost | 0.92 | 0.917 | 0.911 | 0.95 | 0.82 | 0.88 |
| Gradient Boosting | 0.92 | 0.912 | 0.892 | 0.92 | 0.84 | 0.88 |
| Hist Gradient Boosting | 0.93 | 0.919 | 0.917 | 0.94 | 0.86 | 0.90 |
| XGBoost | 0.93 | 0.919 | 0.913 | 0.94 | 0.86 | 0.90 |
| CatBoost | 0.93 | 0.912 | 0.904 | 0.94 | 0.86 | 0.90 |
| LightGBM | 0.93 | 0.912 | 0.911 | 0.95 | 0.86 | 0.90 |

## 🏆 Best Performing Models

The following models achieved the highest performance:

### Top Performers (Accuracy: 93%)
- **Hist Gradient Boosting**
- **XGBoost**
- **CatBoost**
- **LightGBM**

### Key Performance Metrics for Top Models:
- **Accuracy:** 0.93 (93%)
- **ROC-AUC:** 0.919
- **PR-AUC:** 0.917
- **Precision (Diagnosis):** 0.94-0.95
- **Recall (Diagnosis):** 0.86
- **F1-Score (Diagnosis):** 0.90

### Confusion Matrix (XGBoost - Best Performer)
- True Negatives: 274
- False Positives: 8
- False Negatives: 20
- True Positives: 128

## 📉 ROC Curves and PR Curves

ROC curves and Precision-Recall curves were generated for each model, showing:
- Strong performance with ROC-AUC scores above 0.90 for ensemble methods
- PR-AUC scores indicating good precision-recall trade-off

## 🔍 Key Insights

1. **Best Performing Models:** Ensemble methods (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting) significantly outperform simpler models.

2. **Important Features (implied by model performance):**
   - MMSE score
   - Functional Assessment
   - Age
   - Cognitive symptoms (Memory Complaints, Confusion, Disorientation)
   - ADL (Activities of Daily Living)

3. **Model Comparison:**
   - Simple models (Logistic Regression, KNN, Naive Bayes) achieved 60-79% accuracy
   - Ensemble methods achieved 90-93% accuracy
   - Tree-based models showed strong performance

4. **Class Imbalance Consideration:**
   - Training set: 1,719 samples
   - Test set: 430 samples
   - The dataset appears relatively balanced based on performance metrics

## 📋 Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
tqdm
