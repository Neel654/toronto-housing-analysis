# 🏙️ Toronto Housing Market Analysis - End-to-End Data Science & Predictive Modeling Project

A Python-based data science project analyzing Toronto housing prices through exploratory data analysis (EDA), statistical visualization, predictive modeling, and feature importance evaluation using real-world style housing datasets.

[![Python](https://img.shields.io/badge/Python-Data%20Science-blue?style=for-the-badge&logo=python)](https://www.python.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/) [![Machine Learning](https://img.shields.io/badge/ML-Predictive%20Modeling-green?style=for-the-badge)]()

---

## 🎯 Project Overview

Toronto Housing Market Analysis is an end-to-end analytics and machine learning project focused on understanding housing price trends and property valuation factors across Toronto neighborhoods.

The project processes housing datasets through:
- Data loading and cleaning
- Exploratory data analysis (EDA)
- Statistical visualization
- Correlation analysis
- Predictive modeling
- Feature importance evaluation

The repository is structured like a complete data science workflow with separate datasets, generated visualizations, reusable analysis scripts, and modeling outputs.

---

## ✨ Key Capabilities

- ✅ **Exploratory data analysis (EDA)** for housing market trends
- ✅ **Data cleaning and preprocessing** workflows
- ✅ **Visualization pipeline** generating publication-style charts
- ✅ **Correlation analysis** across housing features
- ✅ **Predictive modeling** for housing price estimation
- ✅ **Feature importance evaluation** for explainable insights
- ✅ **Reusable Python analytics workflow** with modular outputs

---

## 🏗️ Architecture & Workflow

```text
┌─────────────────────────┐
│   Raw Housing Dataset   │
│       data/ Layer       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Data Cleaning &        │
│  Preprocessing          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Exploratory Data        │
│ Analysis (EDA)          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Statistical & Visual    │
│ Analysis Layer          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Predictive Modeling &   │
│ Feature Importance      │
└─────────────────────────┘
```

The workflow moves from raw housing data through analysis and visualization into predictive modeling and interpretability-focused outputs.

---

## 📊 Generated Visualizations

The project automatically generates multiple charts and analysis outputs, including:

| Visualization | Description |
|---|---|
| `01_price_distribution.png` | Housing price distribution analysis |
| `02_price_vs_sqft.png` | Price vs square footage relationship |
| `03_correlation_heatmap.png` | Correlation matrix of numeric features |
| `04_neighborhood_prices.png` | Average prices by neighborhood |
| `05_bedrooms_price.png` | Price trends by bedroom count |
| `06_model_predictions.png` | Model predictions vs actual prices |
| `07_feature_importance.png` | Relative importance of predictive features |

Additional generated outputs are stored in the `visualizations/` directory.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python |
| **Data Analysis** | Pandas |
| **Visualization** | Matplotlib |
| **Modeling** | Machine learning workflows |
| **Project Structure** | Modular analytics repository |
| **Dependencies** | Defined in `requirements.txt` |

---

## 📁 Project Structure

```text
toronto-housing-analysis/
├── data/                          # Housing datasets
├── visualizations/                # Generated charts and outputs
├── 01_price_distribution.png      # Distribution analysis
├── 02_price_vs_sqft.png           # Price vs square footage
├── 03_correlation_heatmap.png     # Feature correlation analysis
├── 04_neighborhood_prices.png     # Neighborhood pricing analysis
├── 05_bedrooms_price.png          # Bedroom pricing trends
├── 06_model_predictions.png       # Prediction model results
├── 07_feature_importance.png      # Feature importance analysis
├── toronto_housing_analysis.py    # Main analysis pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

### Core Project Areas

- **`data/`** — Input housing datasets
- **`visualizations/`** — Generated analysis outputs
- **`toronto_housing_analysis.py`** — Main analytics and modeling workflow
- **`requirements.txt`** — Dependency management

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Neel654/toronto-housing-analysis.git
cd toronto-housing-analysis
```

2. **Create a virtual environment**
```bash
python -m venv .venv
```

3. **Activate the environment**

#### Windows
```bash
.venv\Scripts\activate
```

#### macOS/Linux
```bash
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Run the analysis pipeline**
```bash
python toronto_housing_analysis.py
```

The script will:
- Load housing datasets
- Perform EDA workflows
- Generate visualizations
- Train predictive models
- Save analysis outputs automatically

---

## 🔄 Analysis Workflow

### Data Science Lifecycle

1. Housing datasets are loaded from the `data/` directory
2. Data is cleaned and prepared for analysis
3. Exploratory analysis identifies pricing trends
4. Visualizations are generated for insights
5. Predictive models estimate housing prices
6. Feature importance analysis explains model behavior
7. Charts and outputs are exported for review

---

## 📌 Project Highlights

### Data Science Skills
- Exploratory data analysis
- Data preprocessing workflows
- Statistical visualization
- Predictive modeling

### Engineering Focus
- Structured Python project architecture
- Automated chart generation
- Modular analytics pipeline
- Reusable workflow design

### Visualization & Insights
- Correlation heatmaps
- Distribution analysis
- Neighborhood comparisons
- Model explainability outputs

---

## 💡 Why This Project Stands Out

This project demonstrates more than basic notebook experimentation.

It combines:
- End-to-end analytics workflows
- Visualization engineering
- Predictive modeling
- Structured repository organization
- Explainability-focused analysis

The repository reflects practical experience building a complete data science project rather than isolated EDA notebooks.

---

## 🧠 Learning Outcomes

This project demonstrates practical experience with:
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Statistical visualization workflows
- Machine learning model development
- Feature importance analysis
- Python analytics engineering
- End-to-end data science pipelines

---

## 🚀 Future Improvements

Potential future enhancements:
- Interactive dashboards with Plotly or Streamlit
- Advanced regression models
- Geospatial neighborhood mapping
- Real-time housing data ingestion
- Hyperparameter optimization workflows
- Time-series market forecasting
- Deployment as a web analytics platform

---

## 📄 Resume-Ready Description

- Performed an end-to-end data science analysis of Toronto housing data in Python, including preprocessing, visualization, predictive modeling, and feature importance analysis using reusable analytics workflows and generated insights.

---

## 👤 Author

**Neel Prajapati**  
Computer Science Student @ Toronto Metropolitan University

---

⭐ Feel free to explore the repository, contribute improvements, or fork the project!
