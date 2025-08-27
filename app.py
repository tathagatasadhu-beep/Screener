import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
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
    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class FMPDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://urldefense.com/v3/__https://financialmodelingprep.com/api/v3__;!!Dgr3g5d8opDR!WHyZ6Hl65sYcmj4iKdPUKyem293GKpwt1lq5Z031QcHN_XFIa3afchMoOT5MHaLzkFROQZnjeY-zEGuisDIU33Xk1Oo$"

    def get_json(self, endpoint, symbol):
        url = f"{self.base_url}/{endpoint}/{symbol}"
        try:
            response = requests.get(url, params={"apikey": self.api_key})
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}

    def get_comprehensive_data(self, symbol):
        quote = self.get_json("quote", symbol)
        key_metrics = self.get_json("key-metrics", symbol)
        ratios = self.get_json("ratios", symbol)
        profile = self.get_json("profile", symbol)
        enterprise = self.get_json("enterprise-values", symbol)

        if not quote:
            return None

        current_price = quote.get('price', 0)
        market_cap = profile.get('mktCap', 0)
        pe_ratio = key_metrics.get('peRatio') or ratios.get('priceEarningsRatio') or 0
        pb_ratio = key_metrics.get('pbRatio') or ratios.get('priceToBookRatio') or 0
        peg_ratio = key_metrics.get('pegRatio') or 0
        ev_sales = (key_metrics.get('enterpriseValueOverEBITDA') or 0)

        roic = key_metrics.get('roic') or ratios.get('returnOnCapitalEmployed') or 0
        roe = key_metrics.get('roe') or ratios.get('returnOnEquity') or 0
        roa = ratios.get('returnOnAssets') or 0

        debt_to_equity = key_metrics.get('debtToEquity') or ratios.get('debtEquityRatio') or 0
        free_cash_flow = key_metrics.get('freeCashFlowPerShare', 0) * key_metrics.get('numberOfShares', 0) if key_metrics.get('freeCashFlowPerShare') else 0

        wacc = key_metrics.get('wacc') or 8.0  # fallback default

        roic_val = roic * 100 if roic and roic < 1 else roic
        wacc_val = wacc * 100 if wacc and wacc < 1 else wacc
        roic_wacc_spread = roic_val - wacc_val if roic_val and wacc_val else 0

        return {
            'Symbol': symbol,
            'Company Name': profile.get('companyName', symbol),
            'Price ($)': current_price,
            'Market Cap (M)': market_cap / 1_000_000 if market_cap else 0,
            'PE Ratio': pe_ratio,
            'P/B Ratio': pb_ratio,
            'Debt/Equity': debt_to_equity,
            'FCF (M)': free_cash_flow / 1_000_000 if free_cash_flow else 0,
            'PEG Ratio': peg_ratio,
            'EV/Sales': ev_sales,
            'ROIC (%)': roic_val,
            'WACC (%)': wacc_val,
            'ROIC-WACC (%)': roic_wacc_spread,
            'ROE (%)': roe * 100 if roe and roe < 1 else roe,
            'ROA (%)': roa * 100 if roa and roa < 1 else roa
        }

def main():
    st.markdown('<div class="main-header">📊 Stock Screener Pro</div>', unsafe_allow_html=True)

    api_key = st.secrets.get("FMP_API_KEY", "")
    if not api_key:
        st.error("Please add your Financial Modeling Prep API key as FMP_API_KEY in Streamlit Secrets.")
        return

    fetcher = FMPDataFetcher(api_key)

    st.sidebar.header("Configuration")
    stock_input = st.sidebar.text_area("Stock symbols (comma-separated):", value="AAPL,MSFT,GOOGL,AMZN,TSLA")
    symbols = [s.strip().upper() for s in stock_input.split(",") if s.strip()]

    peg_range = st.sidebar.slider("PEG Ratio Range", 0.0, 10.0, (0.0, 5.0), 0.1)
    roic_min = st.sidebar.slider("Minimum ROIC (%)", -100.0, 100.0, -100.0, 1.0)
    market_cap_min = st.sidebar.slider("Minimum Market Cap ($B)", 0.0, 1000.0, 0.0, 1.0)
    fetch_data = st.sidebar.button("Fetch Latest Data")

    if fetch_data or 'stock_data' not in st.session_state:
        st.session_state.stock_data = []
        progress = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(symbols):
            status.text(f"Fetching {sym} ({i+1}/{len(symbols)})...")
            data = fetcher.get_comprehensive_data(sym)
            if 
                st.session_state.stock_data.append(data)
            else:
                st.warning(f"Could not fetch data for {sym}")
            time.sleep(0.3)  # rate limit buffer
            progress.progress((i + 1) / len(symbols))
        status.empty()
        progress.empty()

    if 'stock_data' in st.session_state and st.session_state.stock_
        df = pd.DataFrame(st.session_state.stock_data)
        # Clean data
        for col in ['PE Ratio', 'P/B Ratio', 'Debt/Equity', 'FCF (M)', 'PEG Ratio', 'EV/Sales', 'ROIC (%)', 'WACC (%)', 'ROIC-WACC (%)', 'Market Cap (M)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)

        # Apply filters
        peg_cond = df['PEG Ratio'].between(peg_range[0], peg_range[1], inclusive='both') | df['PEG Ratio'].isna()
        roic_cond = (df['ROIC (%)'] >= roic_min) | df['ROIC (%)'].isna()
        market_cond = (df['Market Cap (M)'] / 1000 >= market_cap_min) | df['Market Cap (M)'].isna()

        filtered_df = df[peg_cond & roic_cond & market_cond]

        st.markdown(f"### Screening Results — {filtered_df.shape[0]} companies match filters:")

        if filtered_df.empty:
            st.warning("No stocks match the current filters.")
        else:
            st.dataframe(filtered_df)

            csv = filtered_df.to_csv(index=False)
            st.download_button("Download CSV", csv, f"stock_screener_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    else:
        st.info("Enter stock symbols and click 'Fetch Latest Data' to begin.")

if __name__ == "__main__":
    main()
