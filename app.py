import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
   
    .screener-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
    }
   
    .metric-tooltip {
        font-size: 0.9rem;
        color: #6c757d;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
   
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
   
    .filter-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
   
    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class FMPDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
       
    def get_quote(self, symbol):
        """Get real-time quote"""
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_key_metrics(self, symbol):
        """Get key financial metrics including ROIC, WACC, PEG"""
        try:
            url = f"{self.base_url}/key-metrics/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_ratios(self, symbol):
        """Get financial ratios"""
        try:
            url = f"{self.base_url}/ratios/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_company_profile(self, symbol):
        """Get company profile"""
        try:
            url = f"{self.base_url}/profile/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_enterprise_values(self, symbol):
        """Get enterprise values for EV/Sales"""
        try:
            url = f"{self.base_url}/enterprise-values/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_comprehensive_data(self, symbol):
        """Get comprehensive financial data for a stock"""
        try:
            # Get all data sources
            quote = self.get_quote(symbol)
            key_metrics = self.get_key_metrics(symbol)
            ratios = self.get_ratios(symbol)
            profile = self.get_company_profile(symbol)
            enterprise = self.get_enterprise_values(symbol)
           
            if not quote:
                return None
           
            # Extract metrics with fallbacks
            current_price = quote.get('price', 0)
            market_cap = profile.get('mktCap', 0)
           
            # Key financial metrics from FMP
            pe_ratio = key_metrics.get('peRatio') or ratios.get('priceEarningsRatio') or 0
            pb_ratio = key_metrics.get('pbRatio') or ratios.get('priceToBookRatio') or 0
            peg_ratio = key_metrics.get('pegRatio') or 0
            ev_sales = key_metrics.get('enterpriseValueOverEBITDA') or enterprise.get('enterpriseValue', 0) / max(key_metrics.get('revenuePerShare', 1) * key_metrics.get('numberOfShares', 1), 1) if key_metrics.get('revenuePerShare') and key_metrics.get('numberOfShares') else 0
           
            # Advanced metrics
            roic = key_metrics.get('roic') or ratios.get('returnOnCapitalEmployed') or 0
            roe = key_metrics.get('roe') or ratios.get('returnOnEquity') or 0
            roa = ratios.get('returnOnAssets') or 0
           
            # Debt and cash flow metrics
            debt_to_equity = key_metrics.get('debtToEquity') or ratios.get('debtEquityRatio') or 0
            current_ratio = ratios.get('currentRatio') or 0
            quick_ratio = ratios.get('quickRatio') or 0
           
            # Cash flow (from key metrics or calculated)
            free_cash_flow = key_metrics.get('freeCashFlowPerShare', 0) * key_metrics.get('numberOfShares', 0) if key_metrics.get('freeCashFlowPerShare') and key_metrics.get('numberOfShares') else 0
           
            # WACC (Financial Modeling Prep provides this in key metrics)
            wacc = key_metrics.get('wacc') or self._estimate_wacc(debt_to_equity, market_cap)
           
            # Calculate ROIC - WACC spread
            roic_val = roic * 100 if roic and roic < 1 else roic if roic else 0  # Convert to percentage if needed
            wacc_val = wacc * 100 if wacc and wacc < 1 else wacc if wacc else 0  # Convert to percentage if needed
            roic_wacc_spread = roic_val - wacc_val if roic_val and wacc_val else 0
           
            return {
                'symbol': symbol,
                'name': profile.get('companyName', symbol),
                'current_price': current_price,
                'market_cap': market_cap / 1_000_000 if market_cap else 0,  # Convert to millions
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'debt_to_equity': debt_to_equity,
                'free_cash_flow': free_cash_flow / 1_000_000 if free_cash_flow else 0,  # Convert to millions
                'peg_ratio': peg_ratio,
                'ev_sales_ratio': ev_sales,
                'roic': roic_val,
                'wacc': wacc_val,
                'roic_wacc_spread': roic_wacc_spread,
                'roe': roe * 100 if roe and roe < 1 else roe if roe else 0,
                'roa': roa * 100 if roa and roa < 1 else roa if roa else 0,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'revenue_growth': key_metrics.get('revenueGrowth', 0) * 100 if key_metrics.get('revenueGrowth') else 0,
                'earnings_growth': key_metrics.get('epsgrowth', 0) * 100 if key_metrics.get('epsgrowth') else 0
            }
           
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
   
    def _estimate_wacc(self, debt_to_equity, market_cap):
        """Estimate WACC if not provided by FMP"""
        try:
            if not debt_to_equity:
                return 8.0  # Default assumption
           
            # Simplified WACC calculation
            risk_free_rate = 4.5  # 10-year treasury
            market_risk_premium = 6.0  # Historical market risk premium
           
            equity_ratio = 1 / (1 + debt_to_equity) if debt_to_equity > 0 else 1
            debt_ratio = 1 - equity_ratio
           
            beta = 1.2  # Default beta
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            cost_of_debt = 5.0  # Default cost of debt
            tax_rate = 25.0  # Default tax rate
           
            wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate/100))
            return wacc
        except:
            return 8.0

def get_metric_tooltip(metric_name):
    """Return tooltip text for each metric"""
    tooltips = {
        'PE Ratio': "A high P/E ratio could mean that a stock's price is high relative to earnings and possibly overvalued. Conversely, a low P/E ratio might indicate that the current stock price is low relative to earnings. Compare PE with Peers",
        'P/B Ratio': "The price-to-book ratio measures whether a stock is over or undervalued by comparing the net value (assets - liabilities) of a company to its market capitalization. A P/B ratio less than 0.5 is a good indication of cheap stock.",
        'Debt/Equity': "A low debt-to-equity ratio means the company uses a lower amount of debt for financing versus shareholder equity.",
        'Free Cash Flow': "Free cash flow can be an early indicator that earnings may increase in the future since increasing free cash flow typically precedes increased earnings.",
        'PEG Ratio': "The PEG ratio measures the relationship between the price/earnings ratio and earnings growth. A stock with a PEG of less than one is considered undervalued.",
        'EV/Sales': "For fast-growing companies that are not yet consistently profitable, the EV/Sales ratio can be used to assess valuation based on revenue. A low EV/Sales ratio suggests the company may be undervalued.",
        'ROIC': "Return on Invested Capital measures how efficiently a company uses its capital. Companies with ROIC > 8% often see significant stock price appreciation.",
        'WACC': "Weighted Average Cost of Capital represents the company's cost of financing. When ROIC exceeds WACC, the company creates value.",
        'ROIC-WACC Spread': "The difference between ROIC and WACC. A positive spread indicates value creation - the higher the spread, the better the company is at generating returns above its cost of capital."
    }
    return tooltips.get(metric_name, "")

def display_metrics_with_tooltips():
    """Display metric explanations with hover tooltips"""
    with st.expander("📊 Metric Explanations", expanded=False):
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("**📈 Valuation Metrics**")
            st.markdown("**P/E Ratio**: " + get_metric_tooltip('PE Ratio'))
            st.markdown("**P/B Ratio**: " + get_metric_tooltip('P/B Ratio'))
            st.markdown("**PEG Ratio**: " + get_metric_tooltip('PEG Ratio'))
            st.markdown("**EV/Sales**: " + get_metric_tooltip('EV/Sales'))
       
        with col2:
            st.markdown("**💰 Financial Health & Efficiency Metrics**")
            st.markdown("**Debt/Equity**: " + get_metric_tooltip('Debt/Equity'))
            st.markdown("**Free Cash Flow**: " + get_metric_tooltip('Free Cash Flow'))
            st.markdown("**ROIC**: " + get_metric_tooltip('ROIC'))
            st.markdown("**WACC**: " + get_metric_tooltip('WACC'))
            st.markdown("**ROIC-WACC Spread**: " + get_metric_tooltip('ROIC-WACC Spread'))

def create_summary_dashboard(df):
    """Create a summary dashboard with key insights"""
    if df.empty:
        return
   
    st.markdown("### 📊 Portfolio Summary")
   
    col1, col2, col3, col4 = st.columns(4)
   
    # Use original column names (before renaming)
    pe_col = 'pe_ratio' if 'pe_ratio' in df.columns else 'PE Ratio'
    pb_col = 'pb_ratio' if 'pb_ratio' in df.columns else 'P/B Ratio'
    peg_col = 'peg_ratio' if 'peg_ratio' in df.columns else 'PEG Ratio'
    roic_col = 'roic' if 'roic' in df.columns else 'ROIC (%)'
    wacc_col = 'wacc' if 'wacc' in df.columns else 'WACC (%)'
   
    with col1:
        pe_data = pd.to_numeric(df[pe_col], errors='coerce')
        pb_data = pd.to_numeric(df[pb_col], errors='coerce')
        undervalued_count = len(df[(pe_data < 15) & (pb_data < 1.5)])
        percentage = f"{undervalued_count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Undervalued Stocks", undervalued_count, percentage)
   
    with col2:
        peg_data = pd.to_numeric(df[peg_col], errors='coerce')
        good_peg_count = len(df[peg_data < 1.0])
        percentage = f"{good_peg_count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Good PEG (<1.0)", good_peg_count, percentage)
   
    with col3:
        roic_data = pd.to_numeric(df[roic_col], errors='coerce')
        wacc_data = pd.to_numeric(df[wacc_col], errors='coerce')
        value_creators = len(df[roic_data > wacc_data])
        percentage = f"{value_creators/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Value Creators", value_creators, percentage)
   
    with col4:
        roic_data = pd.to_numeric(df[roic_col], errors='coerce')
        high_roic_count = len(df[roic_data > 15])
        percentage = f"{high_roic_count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("High ROIC (>15%)", high_roic_count, percentage)

# ... main() and all code up to the last footer section remain unchanged ...

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            📊 Stock Screener Pro | Powered by Financial Modeling Prep API |
            <em>Data is for informational purposes only. Not financial advice.</em>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
