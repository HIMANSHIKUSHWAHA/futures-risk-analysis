# 📊 Futures Market Liquidity Risk Analysis Dashboard

A Streamlit-based interactive dashboard for analyzing liquidity risk in futures markets, integrating advanced risk models, custom metrics, and rich visualizations.

---

## 🚀 Overview

This dashboard enables quantitative analysts, traders, and researchers to:

- Assess **liquidity risk** using advanced market metrics
- Calculate **Value-at-Risk (VaR)** using Historical, GARCH, and EWMA methods
- Estimate **liquidity costs** using trade execution models
- Perform **stress testing** of liquidity shocks
- Analyze **order book depth** (simulated)
- Visualize **market-wide correlations** and **trading insights**

---

## 🔍 Features

- **Real-time interactive selection** of futures contracts and date ranges
- Dynamic KPIs: Bid-Ask Spread, Volume, Illiquidity
- VaR and Liquidity Cost stacked bar visualizations
- Liquidity-adjusted execution cost modeling
- Stress scenarios simulating market shocks
- Historical trends and bubble chart risk matrix
- Correlation heatmaps for systemic risk visibility
- Professional UI with light theme and responsive layout

---

## 📁 Project Structure

```
📦 futures-risk-analysis/
├── app.py                      # Streamlit dashboard code
├── data/
│   ├── raw/                    # Raw CSVs for futures
│   ├── processed/              # Cleaned CSVs
│   └── futures_data.db         # SQLite database
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📦 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/futures-risk-analysis.git
   cd futures-risk-analysis
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:
   - Place raw CSVs (`GC_F.csv`, `CL_F.csv`, `ES_F.csv`) in `data/raw/`
   - Run your data processing script to populate the SQLite database:
     ```bash
     python scripts/load_data.py
     ```

5. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Key Concepts Implemented

- **Bid-Ask Spread (bps)**
- **Amihud Illiquidity Ratio**
- **Market Depth and Resilience**
- **EWMA & GARCH Volatility Modeling**
- **Liquidity-adjusted VaR**
- **Stress Testing Scenarios**
- **TWAP Execution Cost Modeling**
- **Bubble Chart Risk Matrices**

---

## 📸 Sample UI

> Example:  
![Futures Selection](screenshots/futures_selection.png)

---

## 📊 Powered By

- [Streamlit](https://streamlit.io)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)
- [ARCH](https://arch.readthedocs.io/)
- [SQLite](https://sqlite.org/)

---

## 👩‍💻 Author

**Himanshi Kushwaha**  
[LinkedIn](https://www.linkedin.com/in/himanshikushwaha305) | [GitHub](https://github.com/himanshikushwaha305)

---

## 📄 License

MIT License. Feel free to use, extend, and contribute.
