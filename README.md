# Toronto Housing Market Data Science Analysis

Complete end-to-end data science pipeline analyzing Toronto's real estate market using Python, demonstrating practical applications of data cleaning, exploratory analysis, visualization, and machine learning.

## 📊 Project Overview

This project provides a comprehensive analysis of Toronto's housing market through:
- Synthetic dataset generation (1,000+ realistic property listings)
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Advanced visualizations
- Predictive modeling with multiple algorithms
- Feature importance analysis
- Actionable insights for stakeholders

## 🛠️ Technology Stack

- **Python 3.x**: Core programming language
- **pandas & NumPy**: Data manipulation and numerical computing
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning (Linear Regression, Random Forest)
- **Jupyter**: Interactive analysis environment

## 📁 Project Structure
```
toronto-housing-analysis/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── toronto_housing_analysis.py   # Main analysis script
├── notebooks/                     # Jupyter notebooks
├── data/                          # Dataset storage
└── visualizations/                # Generated plots (7 charts)
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/toronto-housing-analysis.git
cd toronto-housing-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Run complete analysis pipeline
python toronto_housing_analysis.py
```

This will generate:
- 7 professional visualizations
- Complete statistical analysis
- Model performance metrics
- Detailed insights report

## 📈 Analysis Pipeline

### 1. Data Generation & Cleaning
- Generate 1,000 synthetic Toronto housing listings
- Validate data integrity (zero missing values)
- Feature engineering (price per sqft)

### 2. Exploratory Data Analysis (EDA)
- Price distribution analysis
- Feature correlations
- Neighborhood comparisons
- Statistical summaries

### 3. Visualization (7 Charts Generated)
1. Price distribution histograms
2. Price vs Square Footage scatter plot
3. Correlation heatmap
4. Neighborhood price rankings
5. Bedroom pricing trends
6. Model prediction accuracy
7. Feature importance chart

### 4. Predictive Modeling
- **Linear Regression**: 76.25% R² accuracy
- **Random Forest**: 87.35% R² accuracy (WINNER!)
- Feature importance analysis
- Model comparison and evaluation

## 🎯 Key Findings

### Price Drivers
1. **Square Footage** (78% importance) - Dominant predictor
2. **Neighborhood** (13% importance) - Significant location premium
3. **Property Age** (4% importance) - Modest negative impact
4. **Bedrooms** (3% importance) - Value addition

### Model Performance
- **Random Forest R²**: 87.35% accuracy
- **Mean Absolute Error**: $137,339
- **RMSE**: $174,454
- Successfully predicts prices based on key features

### Market Statistics
- **Average Price**: $1,189,554
- **Median Price**: $1,134,554
- **Price Range**: $300K - $2.76M
- **Avg Price/Sqft**: $720

### Neighborhood Insights
**Most Expensive (Top 5)**:
1. Forest Hill: $1,458,806
2. Yorkville: $1,430,167
3. Downtown: $1,418,249
4. Rosedale: $1,294,370
5. Leslieville: $1,267,811

## 📊 Generated Visualizations

All charts are automatically saved in high resolution (300 DPI):
- `01_price_distribution.png` - Price and price/sqft histograms
- `02_price_vs_sqft.png` - Scatter plot showing size-price correlation
- `03_correlation_heatmap.png` - Feature correlation matrix
- `04_neighborhood_prices.png` - Ranked neighborhood pricing
- `05_bedrooms_price.png` - Price by bedroom count
- `06_model_predictions.png` - Actual vs predicted prices
- `07_feature_importance.png` - Random Forest feature weights

## 🎓 Educational Value

This project demonstrates:
- End-to-end data science workflow
- Best practices in data analysis
- Machine learning implementation
- Data visualization techniques
- Code documentation and reproducibility
- Real-world application of DS concepts

Perfect for:
- Data science students and learners
- Real estate professionals
- ML practitioners
- Portfolio demonstration
- Interview preparation

## 🔧 Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

## 📝 Dataset

The dataset is **synthetically generated** to:
- Avoid privacy concerns
- Maintain realistic Toronto market patterns
- Enable reproducible research
- Demonstrate data science concepts

**Features**:
- Property ID
- Neighborhood (12 Toronto areas)
- Square footage
- Bedrooms, bathrooms, parking spots
- Property age
- Price and price per sqft

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - free to use for learning and commercial projects

## 👤 Author

Neel Prajapati
- GitHub: [@Neel654](https://github.com/Neel654)

## 🙏 Acknowledgments

- Toronto real estate market data patterns
- scikit-learn documentation
- Data science community best practices

---

**Note**: This project uses synthetic data for educational purposes. For real market analysis, use verified data sources.

## 📧 Contact

Questions? Open an issue or reach out via GitHub!
