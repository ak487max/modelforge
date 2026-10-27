# 🚀 ModelForge — Machine Learning Experimentation Platform

**Train, evaluate, compare, and interpret machine learning models in one interactive platform.**

🔗 **Live Demo:** [Launch ModelForge](https://modelforge-xit9oonprer5mazgqeuyoy.streamlit.app/Prediction_Explainer)  
💻 **GitHub Repository:** [ak487max/modelforge](https://github.com/ak487max/modelforge)

---

## 📌 Overview

ModelForge is an interactive machine learning platform built with Python and Streamlit. It helps users explore datasets, train classification and regression models, evaluate performance, compare experiments, and interpret model predictions.

The goal is to simplify the machine learning workflow by bringing essential model development and evaluation tools into a single application.

## ✨ Features

### 📂 Dataset Studio
- Upload datasets in CSV format.
- Preview and explore dataset contents.
- Select a target column and machine learning task.
- Train classification and regression models.
- Configure the train-test split.

### 📊 Model Evaluation Studio
- View classification metrics, including accuracy, precision, recall, and F1-score.
- Inspect available evaluation results from saved experiments.
- Visualize model performance.

### 🧠 Feature Importance Studio
- Visualize feature importance for supported tree-based models.
- Inspect coefficient magnitudes for supported linear models.
- Download feature rankings as CSV.

### ⚖️ Experiment Comparison
- Compare saved experiments side by side.
- Visualize selected performance metrics.
- Rank experiments using task-appropriate metrics.
- Export comparison results as CSV.

### 🔎 Prediction Explainer
- Generate predictions using saved models.
- Display class probabilities when supported.
- Visualize feature importance.
- Generate SHAP-based explanations for supported tree models.
- Inspect the input data used for a prediction.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning pipelines and evaluation |
| Plotly | Interactive charts and visualizations |
| Joblib | Saving and loading trained models |
| SHAP | Model prediction explanations |

## 🏗️ Project Structure

```text
ModelForge/
├── app.py
├── requirements.txt
├── pages/
│   ├── 1_Dataset_Studio.py
│   ├── 2_Model_Evaluation.py
│   ├── 3_Feature_Importance.py
│   ├── 4_Experiment_Comparison.py
│   └── 5_Prediction_Explainer.py
├── src/
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       ├── storage.py
│       ├── data_loader.py
│       └── experiment_tracker.py
└── artifacts/
    ├── models/
    └── experiments/
```

*The structure above describes the intended organization; adjust the filenames if your repository differs.*

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ak487max/modelforge.git
cd modelforge
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚀 Live Deployment

ModelForge is deployed using Streamlit Community Cloud.

**[Launch the live application →](https://modelforge-xit9oonprer5mazgqeuyoy.streamlit.app/Prediction_Explainer)**

The link opens the Prediction Explainer page. Use the application sidebar to navigate to other available pages.

## 🧪 Example Workflow

1. Upload a CSV dataset in Dataset Studio.
2. Select the target column and task.
3. Train a supported machine learning model.
4. Review evaluation metrics.
5. Compare saved experiments.
6. Inspect feature importance and generate predictions.
7. Use SHAP explanations for supported models to better understand individual predictions.

## 📈 Future Improvements

- More robust dataset validation and preprocessing.
- Additional model algorithms and hyperparameter tuning.
- Improved experiment persistence and model versioning.
- More comprehensive model diagnostics.
- Expanded explainability support for different estimator types.
- Automated testing and improved error handling.

## ⚠️ Limitations

- Available models and explanation methods depend on the saved estimator and its supported capabilities.
- Feature importance does not establish causation.
- SHAP explanations describe model behavior and should be interpreted in the context of the selected model and output.
- Experiment and model persistence depends on the application's storage configuration and deployment environment.

## 👩‍💻 Author

**Akansha Mishra**

B.Tech — Computer Science and Engineering (AI & ML)

**Links**
- [GitHub Profile](https://github.com/ak487max)
- [ModelForge Repository](https://github.com/ak487max/modelforge)
- [Live Demo](https://modelforge-xit9oonprer5mazgqeuyoy.streamlit.app/Prediction_Explainer)

---

*ModelForge is a learning and experimentation platform intended to make machine learning workflows more accessible and interpretable.*
