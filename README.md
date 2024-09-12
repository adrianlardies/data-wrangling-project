# Bitcoin and Gold Investment Analysis Project
<p align="center">
<img src="https://i.giphy.com/94EQmVHkveNck.webp" alt="Investment Graph">
</p>

## Team Members

| Name             | LinkedIn Profile | Brief Description |
|------------------|------------------|-------------------|
| Danny David Rodas Galarza         | https://www.linkedin.com/in/dannyrodasgalarza/      | Data Analyst and Financial Advisor |
| Adrián Lardiés Utrilla         | https://www.linkedin.com/in/adrianlardies/    | Developer and Data Analyst |


## Project Overview

### Business Problem

We are a modern financial advisory startup that specializes in making investment opportunities accessible to the younger generation. Our mission is to simplify investment strategies, particularly for Bitcoin and Gold, empowering individuals with limited financial knowledge or capital to make informed decisions. 

We aim to:
- Make Bitcoin and Gold investments understandable through educational resources and easy-to-read visualizations.
- Showcase how small, consistent investments can yield significant long-term returns.
- Highlight the importance of diversifying assets, using emerging technologies and historical trends in financial markets.

### Initial Hypotheses

1. **Bitcoin as a Growth Asset**: We hypothesize that Bitcoin has shown significant growth potential over time, outperforming other traditional assets such as gold and the S&P 500.
2. **Low-Entry Investment Strategy**: Regular, small investments (such as dollar-cost averaging) in Bitcoin and Gold can provide considerable returns over time.
3. **Market Volatility**: Despite Bitcoin's volatility, a long-term investment strategy can mitigate risk and provide higher potential returns than traditional assets.

### Problem We Are Solving

Many young investors feel overwhelmed by the complexity of cryptocurrency markets, especially Bitcoin. There is a common misconception that large sums of money are required to invest. Our analysis demonstrates that small, consistent investments can yield significant returns, offering an accessible entry point for beginners. We aim to:
- Educate about the accessibility of Bitcoin as a low-barrier, high-potential investment.
- Show how investments in Bitcoin and Gold compare to traditional assets like the S&P 500.

## Data Sources

We will analyze the following datasets to support our hypotheses and provide insights:
- **Historical Bitcoin Prices**: Evolution of Bitcoin prices over time.
- **Historical Gold Prices**: Long-term trends in Gold prices.
- **S&P 500 Index**: Performance of the S&P 500 to compare with cryptocurrency and precious metals.
- **Additional Economic Data**:
    - U.S. Inflation Rate
    - U.S. Federal Reserve Interest Rates
    - Volatility Index (VIX)
    - Global Economic Events and Crises

### API Sources:
- Yahoo Finance (Bitcoin, Gold, and S&P 500 data)
- AlphaVantage (Economic indicators)
- NewsAPI (Economic and global events)

## Methodology

1. **Business Problem and Hypothesis Definition**: Clarify the financial problems we aim to address and outline the hypotheses.
2. **Data Collection and Cleaning**: Extract relevant data from APIs and clean it to handle missing values, remove duplicates, and standardize formats.
3. **Exploratory Data Analysis (EDA)**: Visualize price trends, volatility, and market performance of Bitcoin, Gold, and the S&P 500 over time.
4. **Data Analysis and Hypothesis Testing**: Perform statistical analysis on the performance and volatility of Bitcoin and Gold compared to traditional assets. Test the dollar-cost averaging strategy for different time frames.
5. **Visualization**: Create engaging and easy-to-understand visualizations, including time-series graphs, volatility comparisons, and performance benchmarks.
6. **Conclusion**: Summarize key findings and provide actionable insights for young investors.

## Results

1. **Bitcoin Price Growth**: Over the long term, Bitcoin has shown significant price appreciation compared to traditional assets.
2. **Gold as a Hedge**: While Bitcoin is more volatile, Gold has provided stability as a safe-haven asset during economic downturns.
3. **Dollar-Cost Averaging**: Regular investments in Bitcoin over time could have yielded impressive returns, even with its volatile nature.
4. **Volatility Impact**: While Bitcoin is volatile, strategic long-term investment significantly reduces risk and maximizes returns compared to short-term trades.

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **API Integration**: Yahoo Finance, AlphaVantage, NewsAPI
- **Data Storage**: CSV, xlsx.
- **Version Control**: Git

## How to Use

To replicate or build upon this analysis, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/cohet3/gold_vs_bitcoin.git

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
3. Run the analysis scripts in the src/ directory
     ```bash
   python src/main.ipynb

## Future Work
Advanced Modeling: Implement machine learning models to forecast Bitcoin and Gold prices under different economic scenarios.
Interactive Dashboards: Develop interactive dashboards for young investors to visualize the impact of different investment strategies.
Extended Data Collection: Incorporate more macroeconomic data, such as global crises, geopolitical events, and policy changes, to better understand their effects on asset prices.
## Conclusion
This project highlights the potential of Bitcoin as a high-growth investment and Gold as a stabilizing asset. By presenting accessible data visualizations and investment strategies, we aim to demystify the world of digital currencies and help young investors make informed decisions.