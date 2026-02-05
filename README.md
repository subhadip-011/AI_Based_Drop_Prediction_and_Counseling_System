# 🎓 AI-Based Dropout Prediction and Counseling System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning system that identifies students at risk of dropping out and enables early intervention through predictive analytics and interactive dashboards.

Presented at the **15th Project Innovation Contest (PIC)** alongside **ICDCIT 2026**, KIIT University, Bhubaneswar.

---

## 📌 Project Overview

Student dropout is a critical challenge in educational institutions worldwide. This project leverages **machine learning and predictive analytics** to:

- 🎯 Identify at-risk students before they drop out
- 📊 Analyze patterns in academic, attendance, and socio-economic data
- 🚨 Provide early alerts to counselors and administrators
- 💡 Support data-driven intervention strategies

The system classifies students into three categories: **Dropout**, **Enrolled**, or **Graduate**, achieving **76.9% test accuracy** with a balanced approach across all classes.

---

## ✨ Key Features

- ✅ **Multi-Model Comparison**: Logistic Regression, Random Forest, XGBoost, and Neural Networks
- ✅ **Class Imbalance Handling**: SMOTE-based oversampling for balanced predictions
- ✅ **Robust Evaluation**: 3-fold cross-validation with macro-averaged metrics
- ✅ **Interactive Dashboard**: Real-time risk visualization using Streamlit
- ✅ **Automated Alerts**: Flags high-risk students for immediate counselor review
- ✅ **Production-Ready**: Clean code architecture with modular design

---

## 🎯 Problem Statement

Educational institutions face:
- High dropout rates impacting student success and institutional reputation
- Lack of early warning systems to identify struggling students
- Limited data-driven tools for counselor intervention
- Manual tracking processes that miss early warning signs

**Solution**: An intelligent system that predicts dropout risk and enables proactive intervention.

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Size** | 4,000+ student records |
| **Features** | Academic performance, attendance metrics, socio-economic indicators |
| **Target Classes** | Dropout, Enrolled, Graduate |
| **Data Split** | 80% Training, 20% Testing |

**Note**: Sample/demo data is provided in the repository. Sensitive institutional data has been anonymized or excluded for privacy compliance.

---

## 🛠️ Tech Stack

### Core Technologies
- **Programming**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Database**: SQL (optional for production deployment)

### Key Libraries
```python
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
tensorflow==2.13.0
imbalanced-learn==0.11.0
streamlit==1.28.0
plotly==5.17.0
```

---

## 🔬 Methodology

### 1️⃣ Data Preprocessing
- **Missing Value Imputation**: Statistical methods (mean/median/mode)
- **Feature Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)

### 2️⃣ Exploratory Data Analysis (EDA)
- Distribution analysis of target classes
- Correlation heatmaps for feature relationships
- Identification of key dropout indicators
- Outlier detection and treatment

### 3️⃣ Model Development

| Model | Description | Use Case |
|-------|-------------|----------|
| **Logistic Regression** | Baseline linear classifier | Quick baseline performance |
| **Random Forest** | Ensemble tree-based model | Feature importance analysis |
| **XGBoost** | Gradient boosting classifier | Best generalization performance |
| **Artificial Neural Network** | Deep learning model | Complex pattern recognition |

### 4️⃣ Model Evaluation

**Evaluation Strategy**:
- 3-fold Cross-Validation for robust performance estimation
- Stratified sampling to maintain class distribution

**Metrics**:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score (Macro-averaged)
- ✅ Confusion Matrix
- ✅ ROC-AUC Score

### 5️⃣ Model Selection
**Winner**: **XGBoost Classifier**
- Achieved **76.9% test accuracy**
- Balanced macro F1-score across all classes
- Superior generalization on unseen data
- **~12% improvement** over baseline through hyperparameter tuning

---

## 📈 Results & Performance

### Model Comparison

| Model | Test Accuracy | Macro F1-Score | Training Time |
|-------|---------------|----------------|---------------|
| Logistic Regression | 68.3% | 0.65 | Fast |
| Random Forest | 73.5% | 0.71 | Moderate |
| **XGBoost** | **76.9%** | **0.75** | Moderate |
| Neural Network (ANN) | 75.2% | 0.73 | Slow |

### Key Insights
- 📌 Attendance rate is the strongest predictor of dropout
- 📌 Socio-economic factors significantly influence outcomes
- 📌 Early semester performance correlates with final status
- 📌 Combined features outperform individual indicators

---

## 🖥️ Dashboard Features

The **Streamlit dashboard** provides:

1. **Risk Overview**: Summary statistics and risk distribution
2. **Student Search**: Look up individual student predictions
3. **Risk Filters**: Filter by high/medium/low risk categories
4. **Visualizations**:
   - Risk distribution charts
   - Feature importance plots
   - Trend analysis over time
5. **Alert System**: Automated notifications for high-risk students
6. **Export Functionality**: Download reports for counselor review

---

## 📁 Project Structure

```
AI-Based-Dropout-Prediction/
│
├── data/
│   ├── raw/                    # Original datasets (not included)
│   ├── processed/              # Cleaned and preprocessed data
│   └── demo_data.csv           # Sample data for demonstration
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data cleaning and transformation
│   └── 03_Modeling.ipynb      # Model training and evaluation
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning utilities
│   ├── feature_engineering.py # Feature creation functions
│   ├── model_training.py      # ML model training pipeline
│   └── evaluation.py          # Model evaluation metrics
│
├── app/
│   ├── app.py                 # Streamlit dashboard
│   ├── utils.py               # Helper functions for app
│   └── assets/                # Images, CSS, etc.
│
├── models/
│   └── .gitkeep              # Placeholder (models excluded)
│
├── config/
│   └── config.yaml           # Configuration parameters
│
├── tests/
│   └── test_preprocessing.py # Unit tests
│
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
└── README.md               # This file
```

---

## ▶️ Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/subhadip-011/AI-Based-Dropout-Prediction.git
cd AI-Based-Dropout-Prediction
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train the Model
```bash
python src/model_training.py
```

This will:
- Load and preprocess the data
- Train all models with cross-validation
- Save the best model (XGBoost) to `models/` directory
- Generate evaluation reports

#### 2. Run the Dashboard
```bash
streamlit run app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### 3. Make Predictions
```python
import pickle
import pandas as pd

# Load the trained model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new student data
new_student = pd.DataFrame({
    'attendance_rate': [0.75],
    'gpa': [2.8],
    'age': [19],
    # ... other features
})

# Predict
prediction = model.predict(new_student)
print(f"Predicted Status: {prediction[0]}")
```

---

## 🧪 Model Training Details

### Hyperparameter Tuning (XGBoost)

```python
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss'
}
```

### Cross-Validation Strategy
- **Method**: Stratified K-Fold (K=3)
- **Purpose**: Ensure balanced class distribution in each fold
- **Metric**: Macro F1-Score for multi-class evaluation

---

## 📚 Key Learnings

### Technical Insights
✅ Handling severe class imbalance in real-world educational data  
✅ Importance of macro-averaged metrics for balanced evaluation  
✅ Feature engineering significantly impacts model performance  
✅ Ensemble methods (XGBoost) often outperform single models  
✅ Cross-validation prevents overfitting and ensures generalization

### Domain Insights
✅ Early intervention is most effective in first semester  
✅ Attendance is a stronger predictor than grades alone  
✅ Socio-economic support improves retention rates  
✅ Dashboard usability is critical for counselor adoption

---

## 🚀 Future Enhancements

- [ ] **Real-time Integration**: Connect with institutional databases
- [ ] **Advanced NLP**: Analyze student feedback and sentiment
- [ ] **Mobile App**: Enable counselors to receive alerts on mobile
- [ ] **Explainable AI**: SHAP/LIME for model interpretability
- [ ] **Longitudinal Analysis**: Track student progress over multiple semesters
- [ ] **API Development**: REST API for third-party integrations
- [ ] **Multi-institutional**: Generalize model across different institutions

---

## 🏆 Recognition & Achievements

**15th Project Innovation Contest (PIC)**  
*22nd International Conference on Distributed Computing and Intelligent Technology (ICDCIT 2026)*  
📍 KIIT Deemed to be University, Bhubaneswar

This project was selected for presentation among numerous submissions, demonstrating its innovation and practical applicability in educational analytics.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Subhadip Pan**  
Data Science & Analytics Aspirant

- 📧 Email: [Pansubhadip779@gmail.com](mailto:Pansubhadip779@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/subhadip-pan-62b194260](https://www.linkedin.com/in/subhadip-pan-62b194260)
- 🐙 GitHub: [@subhadip-011](https://github.com/subhadip-011)
- 📱 Phone: +91 8116368029

---

## 🙏 Acknowledgments

- KIIT University for hosting the Project Innovation Contest
- Faculty advisors for guidance and mentorship
- Open-source community for amazing tools and libraries
- Educational institutions for highlighting this critical problem

---

## 📞 Support

If you find this project helpful, please ⭐ star the repository!

For questions or suggestions:
- Open an [Issue](https://github.com/subhadip-011/AI-Based-Dropout-Prediction/issues)
- Email: Pansubhadip779@gmail.com

---

## 📖 Citations

If you use this project in your research or work, please cite:

```bibtex
@software{Pan2026DropoutPrediction,
  author = {Pan, Subhadip},
  title = {AI-Based Dropout Prediction and Counseling System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/subhadip-011/AI-Based-Dropout-Prediction}
}
```

---

<div align="center">

**Made with ❤️ and Python**

*Empowering educators with data-driven insights to improve student success*

</div>
