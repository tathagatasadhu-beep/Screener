import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import plotly.express as px
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
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_key_metrics(self, symbol):
        try:
            url = f"{self.base_url}/key-metrics/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_ratios(self, symbol):
        try:
            url = f"{self.base_url}/ratios/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_company_profile(self, symbol):
        try:
            url = f"{self.base_url}/profile/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_enterprise_values(self, symbol):
        try:
            url = f"{self.base_url}/enterprise-values/{symbol}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data[0] if response.status_code == 200 and data else {}
        except:
            return {}
   
    def get_comprehensive_data(self, symbol):
        try:
            quote = self.get_quote(symbol)
            key_metrics = self.get_key_metrics(symbol)
            ratios = self.get_ratios(symbol)
            profile = self.get_company_profile(symbol)
            enterprise = self.get_enterprise_values(symbol)

            if not quote:
                return None

            current_price = quote.get('price', 0)
            market_cap = profile.get('mktCap', 0)
            pe_ratio = key_metrics.get('peRatio') or ratios.get('priceEarningsRatio') or 0
            pb_ratio = key_metrics.get('pbRatio') or ratios.get('priceToBookRatio') or 0
            peg_ratio = key_metrics.get('pegRatio') or 0
            ev_sales = key_metrics.get('enterpriseValueOverEBITDA') or enterprise.get('enterpriseValue', 0) / max(key_metrics.get('revenuePerShare', 1) * key_metrics.get('numberOfShares', 1), 1) if key_metrics.get('revenuePerShare') and key_metrics.get('numberOfShares') else 0

            roic = key_metrics.get('roic') or ratios.get('returnOnCapitalEmployed') or 0
            roe = key_metrics.get('roe') or ratios.get('returnOnEquity') or 0
            roa = ratios.get('returnOnAssets') or 0

            debt_to_equity = key_metrics.get('debtToEquity') or ratios.get('debtEquityRatio') or 0
            current_ratio = ratios.get('currentRatio') or 0
            quick_ratio = ratios.get('quickRatio') or 0

            free_cash_flow = key_metrics.get('freeCashFlowPerShare', 0) * key_metrics.get('numberOfShares', 0) if key_metrics.get('freeCashFlowPerShare') and key_metrics.get('numberOfShares') else 0

            wacc = key_metrics.get('wacc') or self._estimate_wacc(debt_to_equity, market_cap)

            roic_val = roic * 100 if roic and roic < 1 else roic if roic else 0
            wacc_val = wacc * 100 if wacc and wacc < 1 else wacc if wacc else 0
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
        try:
            if not debt_to_equity:
                return 8.0
            risk_free_rate = 4.5
            market_risk_premium = 6.0
            equity_ratio = 1 / (1 + debt_to_equity) if debt_to_equity > 0 else 1
            debt_ratio = 1 - equity_ratio
            beta = 1.2
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            cost_of_debt = 5.0
            tax_rate = 25.0
            wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate/100))
            return wacc
        except:
            return 8.0


def get_metric_tooltip(metric_name):
    tooltips = {
        'PE Ratio': "A high P/E ratio could mean that a stock's price is high relative to earnings and possibly overvalued. Conversely, a low P/E ratio might indicate that the current stock price is low relative to earnings.",
        'P/B Ratio': "Price-to-book ratio compares the company's market value to its book value.",
        'Debt/Equity': "Low debt-to-equity suggests less leverage and lower risk.",
        'Free Cash Flow': "A strong free cash flow supports growth and dividends.",
        'PEG Ratio': "PEG below 1 can mean a stock is undervalued based on expected growth.",
        'EV/Sales': "EV/Sales helps value non-profitable growth companies on revenue.",
        'ROIC': "Return on Invested Capital: Above 8% is generally good.",
        'WACC': "Weighted Average Cost of Capital: the company's blended cost of capital.",
        'ROIC-WACC Spread': "How much is ROIC above the cost of capital? Positive spread is value creation."
    }
    return tooltips.get(metric_name, "")


def display_metrics_with_tooltips():
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
    if df.empty:
        return
    st.markdown("### 📊 Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
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

def main():
    st.markdown('<div class="main-header">📊 Stock Screener Pro</div>', unsafe_allow_html=True)
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Undervalued Compounders"

    tabs = ["Undervalued Compounders", "Growth Stocks", "Dividend Aristocrats", "Small Cap Gems"]
    selected_tab = st.selectbox("Select Screener:", tabs, index=0)
    api_key = st.secrets.get("FMP_API_KEY", "")

    if not api_key:
        st.error("🔑 Financial Modeling Prep API key not found. Please add FMP_API_KEY to your Streamlit secrets.")
        st.info("""
        **How to get FMP API Key:**
        1. Go to [financialmodelingprep.com](https://financialmodelingprep.com)
        2. Sign up for free account (250 calls/day)
        3. Get your API key from dashboard
        4. Add it to Streamlit Secrets as FMP_API_KEY
        """)
        st.stop()

    fetcher = FMPDataFetcher(api_key)

    if selected_tab == "Undervalued Compounders":
        st.markdown("""
        <div class="screener-header">
            <strong>Screener #1: Undervalued Compounders</strong><br>
            <em>Companies that are both <strong>cheap</strong> and <strong>growing</strong></em>
        </div>
        """, unsafe_allow_html=True)

        display_metrics_with_tooltips()

        with st.sidebar:
            st.header("🔧 Configuration")
            st.subheader("📈 Stock Symbols")
            default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,JNJ,V,WMT,PG,UNH,HD,MA,BAC,ABBV,LLY,AVGO,XOM,CVX,PFE,TMO,COST,NFLX,CRM,ABT,ADBE,MRK,ACN,CSCO,PEP,NKE,LIN,DHR,VZ,CMCSA,QCOM,TXN,NEE,HON,UPS,LOW,PM,RTX,BMY,UNP,AMGN,T,SPGI,GS,MDT,CAT,ISRG,GILD,CVS,BLK,MMM,AMT,SYK,LRCX,MU,ELV,CI,ZTS,CB,DE,SO,DUK,BSX,ITW,APD,AON,CL,EMR,NSC,MMC,GD,PGR,EQIX,ICE,HUM,TFC,ETN,FIS,USB,PNC,GM,F,EBAY,PYPL"
            stock_input = st.text_area(
                "Enter stock symbols (comma-separated):",
                value=default_stocks,
                height=100,
                help="Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)"
            )
            symbols = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
            st.subheader("🎛️ Filters")
            peg_range = st.slider("PEG Ratio Range", 0.0, 10.0, (0.0, 5.0), 0.1, help="PEG < 1.0 typically indicates undervalued growth")
            roic_min = st.slider("Minimum ROIC (%)", -100.0, 100.0, -100.0, 1.0, help="No restriction by default")
            market_cap_min = st.slider("Minimum Market Cap ($B)", 0.0, 100.0, 0.0, 1.0, help="0B+ shows all companies")
            show_debug = st.checkbox("🔍 Show Filter Debug Info", value=True)
            fetch_data = st.button("🔄 Fetch Latest Data", type="primary")

        if fetch_data or 'stock_data' not in st.session_state:
            with st.spinner("📡 Fetching comprehensive data from Financial Modeling Prep..."):
                stock_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, symbol in enumerate(symbols):
                    status_text.text(f"Fetching data for {symbol}... ({i+1}/{len(symbols)})")
                    data = fetcher.get_comprehensive_data(symbol)
                    if stock_data.append(data)
                    else:
                        st.warning(f"⚠️ Could not fetch data for {symbol}")
                    time.sleep(0.3)
                    progress_bar.progress((i + 1) / len(symbols))
                st.session_state.stock_data = stock_data
                progress_bar.empty()
                status_text.empty()
                if stock_
                    st.success(f"✅ Successfully fetched data for {len(stock_data)} companies!")
                else:
                    st.error("❌ No data was fetched. Please check your API key and try again.")

        if 'stock_data' in st.session_state and st.session_state.stock_
            df = pd.DataFrame(st.session_state.stock_data)
            numeric_columns = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'free_cash_flow',
                                'peg_ratio', 'ev_sales_ratio', 'roic', 'wacc', 'roic_wacc_spread', 'market_cap']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            create_summary_dashboard(df)
            display_df = df.copy()
            display_df = display_df.rename(columns={
                'symbol': 'Symbol',
                'name': 'Company Name',
                'current_price': 'Price ($)',
                'market_cap': 'Market Cap (M)',
                'pe_ratio': 'PE Ratio',
                'pb_ratio': 'P/B Ratio',
                'debt_to_equity': 'Debt/Equity',
                'free_cash_flow': 'FCF (M)',
                'peg_ratio': 'PEG Ratio',
                'ev_sales_ratio': 'EV/Sales',
                'roic': 'ROIC (%)',
                'wacc': 'WACC (%)',
                'roic_wacc_spread': 'ROIC-WACC (%)'
            })
            for col in ['Market Cap (M)', 'FCF (M)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:,.0f}" if pd.notnull(x) and x != 0 else "N/A"
                    )
            if 'Price ($)' in display_df.columns:
                display_df['Price ($)'] = display_df['Price ($)'].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) and x != 0 else "N/A"
                )
            market_cap_billions = pd.to_numeric(
                display_df['Market Cap (M)'].str.replace(',', '').str.replace('N/A', '0'),
                errors='coerce'
            ) / 1000
            peg_values = pd.to_numeric(display_df['PEG Ratio'], errors='coerce')
            roic_values = pd.to_numeric(display_df['ROIC (%)'], errors='coerce')
            peg_condition = (
                peg_values.between(peg_range[0], peg_range[1], inclusive='both') |
                peg_values.isna() |
                (peg_values == 0)
            )
            roic_condition = (roic_values >= roic_min) | roic_values.isna()
            market_cap_condition = (market_cap_billions >= market_cap_min) | market_cap_billions.isna()
            filtered_df = display_df[peg_condition & roic_condition & market_cap_condition]
            if show_debug:
                st.markdown("### 🔍 Filter Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Companies", len(display_df))
                    st.metric("Pass PEG Filter", peg_condition.sum())
                    st.metric("Pass ROIC Filter", roic_condition.sum())
                    st.metric("Pass Market Cap Filter", market_cap_condition.sum())
                with col2:
                    st.markdown("**🚀 Best Value Creators (ROIC-WACC):**")
                    high_spread_df = filtered_df[pd.to_numeric(filtered_df['ROIC-WACC (%)'], errors='coerce') > 5].head(3)
                    if not high_spread_df.empty:
                        for _, row in high_spread_df.iterrows():
                            roic_wacc = row['ROIC-WACC (%)']
                            if pd.notnull(roic_wacc):
                                st.markdown(f"• **{row['Symbol']}** - Spread: {roic_wacc:.1f}%")
                            else:
                                st.markdown(f"• **{row['Symbol']}** - Spread: N/A")
                    else:
                        st.markdown("No stocks with ROIC-WACC > 5% found")
                with col3:
                    all_conditions = peg_condition & roic_condition & market_cap_condition
                    st.metric("Pass All Filters", all_conditions.sum())
                    peg_available = (~peg_values.isna() & (peg_values != 0)).sum()
                    roic_available = (~roic_values.isna() & (roic_values != 0)).sum()
                    st.metric("PEG Data Available", peg_available, f"{peg_available/len(display_df)*100:.1f}%")
                    st.metric("ROIC Data Available", roic_available, f"{roic_available/len(display_df)*100:.1f}%")

            st.markdown("### 📋 Screening Results")
            st.markdown(f"**{len(filtered_df)}** companies match your criteria out of **{len(display_df)}** analyzed")

            if len(filtered_df) == 0 and len(display_df) > 0:
                st.warning("⚠️ No companies pass the current filters.")
                st.markdown("**📊 Sample of available data (first 10 companies):**")
                sample_cols = ['Symbol', 'Company Name', 'PEG Ratio', 'ROIC (%)', 'WACC (%)', 'ROIC-WACC (%)', 'Market Cap (M)']
                available_cols = [col for col in sample_cols if col in display_df.columns]
                st.dataframe(display_df[available_cols].head(10), use_container_width=True)

            if not filtered_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    sort_column = st.selectbox("Sort by:",
                        ['PEG Ratio', 'ROIC (%)', 'ROIC-WACC (%)', 'WACC (%)', 'PE Ratio', 'P/B Ratio', 'Market Cap (M)'])
                with col2:
                    sort_order = st.selectbox("Order:", ['Ascending', 'Descending'])
                sort_ascending = sort_order == 'Ascending'
                if sort_column in ['Market Cap (M)', 'FCF (M)']:
                    temp_col = pd.to_numeric(
                        filtered_df[sort_column].str.replace(',', '').str.replace('N/A', '0'),
                        errors='coerce'
                    )
                    sorted_indices = temp_col.argsort()
                    if not sort_ascending:
                        sorted_indices = sorted_indices[::-1]
                    filtered_df = filtered_df.iloc[sorted_indices]
                else:
                    filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')
                st.markdown('<div class="data-table">', unsafe_allow_html=True)
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"undervalued_compounders_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.markdown("### 💡 Investment Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Best PEG Opportunities:**")
                    low_peg_df = filtered_df[pd.to_numeric(filtered_df['PEG Ratio'], errors='coerce') < 1.0].head(3)
                    if not low_peg_df.empty:
                        for _, row in low_peg_df.iterrows():
                            st.markdown(f"• **{row['Symbol']}** - PEG: {row['PEG Ratio']}")
                    else:
                        st.markdown("No stocks with PEG < 1.0 found")
                with col2:
                    st.markdown("**🏆 Top ROIC Stocks:**")
                    top_roic_df = filtered_df[pd.to_numeric(filtered_df['ROIC (%)'], errors='coerce') > 15].head(3)
                    if not top_roic_df.empty:
                        for _, row in top_roic_df.iterrows():
                            st.markdown(f"• **{row['Symbol']}** - ROIC: {row['ROIC (%)']}")
                    else:
                        st.markdown("No stocks with ROIC > 15% found")
        else:
            st.info("👆 Click 'Fetch Latest Data' to start screening stocks!")
    else:
        st.info(f"🚧 {selected_tab} screener is coming soon! Stay tuned for more screening strategies.")

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
