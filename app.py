import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Stock Screener Pro", page_icon="📊", layout="wide")

class FMPDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

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

        if not quote:
            return None

        return {
            'Symbol': symbol,
            'Company Name': profile.get('companyName', symbol),
            'Price ($)': quote.get('price', 0),
            'Market Cap (M)': profile.get('mktCap', 0) / 1e6 if profile.get('mktCap') else 0,
            'PE Ratio': key_metrics.get('peRatio') or ratios.get('priceEarningsRatio') or 0,
            'P/B Ratio': key_metrics.get('pbRatio') or ratios.get('priceToBookRatio') or 0,
            'Debt/Equity': key_metrics.get('debtToEquity') or ratios.get('debtEquityRatio') or 0,
            'FCF (M)': (key_metrics.get('freeCashFlowPerShare', 0) * key_metrics.get('numberOfShares', 0)) / 1e6 if key_metrics.get('freeCashFlowPerShare') else 0,
            'PEG Ratio': key_metrics.get('pegRatio') or 0,
            'EV/Sales': key_metrics.get('enterpriseValueOverEBITDA') or 0,
            'ROIC (%)': (key_metrics.get('roic') or ratios.get('returnOnCapitalEmployed') or 0) * 100,
            'WACC (%)': (key_metrics.get('wacc') or 8.0) * 100,
            'ROIC-WACC (%)': ((key_metrics.get('roic') or 0) - (key_metrics.get('wacc') or 0)) * 100,
        }

def main():
    st.title("📊 Stock Screener Pro")

    api_key = st.secrets.get("FMP_API_KEY", "")
    if not api_key:
        st.error("Add your FMP API key to Streamlit secrets as FMP_API_KEY.")
        return

    fetcher = FMPDataFetcher(api_key)

    with st.sidebar:
        st.header("Settings")
        symbols_raw = st.text_area("Enter stock symbols (comma separated):", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        peg_range = st.slider("PEG Ratio Range", 0.0, 10.0, (0.0, 2.0))
        roic_min = st.slider("Minimum ROIC (%)", -100.0, 100.0, 0.0)
        market_cap_min = st.slider("Minimum Market Cap ($B)", 0.0, 500.0, 0.0)
        fetch_button = st.button("Fetch Latest Data")

    if fetch_button or 'stock_data' not in st.session_state:
        st.session_state.stock_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, sym in enumerate(symbols):
            status_text.text(f"Fetching data for {sym} ({i+1}/{len(symbols)})...")
            data = fetcher.get_comprehensive_data(sym)
            if data:
                st.session_state.stock_data.append(data)
            else:
                st.warning(f"Failed to fetch data for {sym}")
            time.sleep(0.3)
            progress_bar.progress((i + 1) / len(symbols))

        status_text.empty()
        progress_bar.empty()

    if 'stock_data' in st.session_state and st.session_state.stock_data:
        df = pd.DataFrame(st.session_state.stock_data)

        # Filters
        df['PEG Ratio'] = pd.to_numeric(df['PEG Ratio'], errors='coerce')
        df['ROIC (%)'] = pd.to_numeric(df['ROIC (%)'], errors='coerce')
        df['Market Cap (M)'] = pd.to_numeric(df['Market Cap (M)'], errors='coerce')
        df['Market Cap (B)'] = df['Market Cap (M)'] / 1000

        filtered = df[
            df['PEG Ratio'].between(peg_range[0], peg_range[1], inclusive='both') &
            (df['ROIC (%)'] >= roic_min) &
            (df['Market Cap (B)'] >= market_cap_min)
        ]

        st.subheader(f"Filtered results: {len(filtered)} of {len(df)} companies")

        st.dataframe(filtered)

        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, "stock_screener.csv", "text/csv")
    else:
        st.info("Enter symbols and click 'Fetch Latest Data'.")

if __name__ == "__main__":
    main()
