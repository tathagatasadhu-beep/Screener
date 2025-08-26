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

class FinnhubDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
    def get_company_profile(self, symbol):
        """Get company profile data"""
        try:
            url = f"{self.base_url}/stock/profile2"
            params = {"symbol": symbol, "token": self.api_key}
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_basic_financials(self, symbol):
        """Get basic financial metrics"""
        try:
            url = f"{self.base_url}/stock/metric"
            params = {"symbol": symbol, "metric": "all", "token": self.api_key}
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_quote(self, symbol):
        """Get real-time quote"""
        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "token": self.api_key}
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_company_financials(self, symbol):
        """Get comprehensive financial data"""
        try:
            # Get basic financials
            basic_financials = self.get_basic_financials(symbol)
            quote = self.get_quote(symbol)
            profile = self.get_company_profile(symbol)
            
            if not basic_financials.get('metric'):
                return None
            
            metrics = basic_financials.get('metric', {})
            current_price = quote.get('c', 0)
            
            # Calculate missing metrics if data is available
            market_cap = profile.get('marketCapitalization', 0)
            shares_outstanding = profile.get('shareOutstanding', 0)
            
            return {
                'symbol': symbol,
                'name': profile.get('name', symbol),
                'current_price': current_price,
                'market_cap': market_cap,
                'pe_ratio': metrics.get('peBasicExclExtraTTM', 0),
                'pb_ratio': metrics.get('pbQuarterly', 0),
                'debt_to_equity': metrics.get('totalDebt2EquityQuarterly', 0),
                'free_cash_flow': metrics.get('freeCashFlowTTM', 0),
                'peg_ratio': metrics.get('pegRatio', 0),
                'ev_sales_ratio': metrics.get('evSalesTTM', 0),
                'roic': metrics.get('roicTTM', 0),
                'revenue_growth': metrics.get('revenueGrowthTTM', 0),
                'earnings_growth': metrics.get('epsGrowthTTM', 0),
                'gross_margin': metrics.get('grossMarginTTM', 0),
                'operating_margin': metrics.get('operatingMarginTTM', 0),
                'net_margin': metrics.get('netMarginTTM', 0),
                'current_ratio': metrics.get('currentRatioQuarterly', 0),
                'quick_ratio': metrics.get('quickRatioQuarterly', 0),
                'roe': metrics.get('roeTTM', 0),
                'roa': metrics.get('roaTTM', 0)
            }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

def calculate_wacc_estimate(data):
    """Calculate a simplified WACC estimate"""
    try:
        # Simplified WACC calculation (this is an approximation)
        # In reality, WACC requires more detailed financial data
        risk_free_rate = 0.045  # Approximate 10-year treasury rate
        market_risk_premium = 0.06  # Historical market risk premium
        
        # Use debt-to-equity to estimate cost of capital
        debt_to_equity = data.get('debt_to_equity', 0)
        equity_ratio = 1 / (1 + debt_to_equity) if debt_to_equity > 0 else 1
        debt_ratio = 1 - equity_ratio
        
        # Estimate cost of equity (simplified CAPM)
        beta = 1.2  # Default beta assumption
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        
        # Estimate cost of debt
        cost_of_debt = 0.05  # Default assumption
        tax_rate = 0.25  # Default tax rate
        
        # Calculate WACC
        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
        return wacc * 100  # Return as percentage
    except:
        return 8.0  # Default WACC assumption

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
        'WACC': "Weighted Average Cost of Capital represents the company's cost of financing. When ROIC exceeds WACC, the company creates value."
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
            st.markdown("**💰 Financial Health Metrics**")
            st.markdown("**Debt/Equity**: " + get_metric_tooltip('Debt/Equity'))
            st.markdown("**Free Cash Flow**: " + get_metric_tooltip('Free Cash Flow'))
            st.markdown("**ROIC**: " + get_metric_tooltip('ROIC'))
            st.markdown("**WACC**: " + get_metric_tooltip('WACC'))

def create_summary_dashboard(df):
    """Create a summary dashboard with key insights"""
    if df.empty:
        return
    
    st.markdown("### 📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Use original column names (before renaming)
    pe_col = 'pe_ratio' if 'pe_ratio' in df.columns else 'PE Ratio'
    pb_col = 'pb_ratio' if 'pb_ratio' in df.columns else 'P/B Ratio'
    roic_col = 'roic' if 'roic' in df.columns else 'ROIC (%)'
    wacc_col = 'wacc' if 'wacc' in df.columns else 'WACC (%)'
    
    with col1:
        pe_data = pd.to_numeric(df[pe_col], errors='coerce')
        pb_data = pd.to_numeric(df[pb_col], errors='coerce')
        undervalued_count = len(df[(pe_data < 15) & (pb_data < 1.5)])
        percentage = f"{undervalued_count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Undervalued Stocks", undervalued_count, percentage)
    
    with col2:
        roic_data = pd.to_numeric(df[roic_col], errors='coerce')
        high_roic_count = len(df[roic_data > 8])
        percentage = f"{high_roic_count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("High ROIC (>8%)", high_roic_count, percentage)
    
    with col3:
        roic_data = pd.to_numeric(df[roic_col], errors='coerce')
        wacc_data = pd.to_numeric(df[wacc_col], errors='coerce')
        value_creators = len(df[roic_data > wacc_data])
        percentage = f"{value_creators/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Value Creators", value_creators, percentage)
    
    with col4:
        pe_data = pd.to_numeric(df[pe_col], errors='coerce')
        avg_pe = pe_data.replace([np.inf, -np.inf], np.nan).dropna().mean()
        avg_pe_display = f"{avg_pe:.1f}" if pd.notnull(avg_pe) else "N/A"
        st.metric("Avg P/E Ratio", avg_pe_display)

def main():
    # App header
    st.markdown('<div class="main-header">📊 Stock Screener Pro</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Undervalued Compounders"
    
    # Tab selection (for future expansion)
    tabs = ["Undervalued Compounders", "Growth Stocks", "Dividend Aristocrats", "Small Cap Gems"]
    selected_tab = st.selectbox("Select Screener:", tabs, index=0)
    
    # API Key input
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
    
    if not api_key:
        st.error("🔑 Finnhub API key not found. Please add FINNHUB_API_KEY to your Streamlit secrets.")
        st.info("Go to Streamlit Cloud → Settings → Secrets to add your API key.")
        st.stop()
    
    # Initialize data fetcher
    fetcher = FinnhubDataFetcher(api_key)
    
    if selected_tab == "Undervalued Compounders":
        st.markdown("""
        <div class="screener-header">
            <strong>Screener #1: Undervalued Compounders</strong><br>
            <em>Companies that are both <strong>cheap</strong> and <strong>growing</strong></em>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metric explanations
        display_metrics_with_tooltips()
        
        # Sidebar for stock selection and filters
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Stock symbols input
            st.subheader("📈 Stock Symbols")
            default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,JNJ,V"
            stock_input = st.text_area(
                "Enter stock symbols (comma-separated):",
                value=default_stocks,
                height=100,
                help="Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)"
            )
            
            # Parse stock symbols
            symbols = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
            
            st.subheader("🎛️ Filters")
            
            # Filter controls
            pe_range = st.slider("P/E Ratio Range", 0.0, 50.0, (0.0, 30.0), 0.5)
            pb_range = st.slider("P/B Ratio Range", 0.0, 10.0, (0.0, 3.0), 0.1)
            roic_min = st.slider("Minimum ROIC (%)", -20.0, 50.0, 0.0, 1.0)
            
            # Fetch data button
            fetch_data = st.button("🔄 Fetch Latest Data", type="primary")
        
        # Main content area
        if fetch_data or 'stock_data' not in st.session_state:
            with st.spinner("📡 Fetching real-time data from Finnhub..."):
                stock_data = []
                progress_bar = st.progress(0)
                
                for i, symbol in enumerate(symbols):
                    data = fetcher.get_company_financials(symbol)
                    if data:
                        # Add calculated WACC
                        data['wacc'] = calculate_wacc_estimate(data)
                        stock_data.append(data)
                    
                    # Add delay to respect API rate limits
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) / len(symbols))
                
                st.session_state.stock_data = stock_data
                progress_bar.empty()
        
        # Display results
        if 'stock_data' in st.session_state and st.session_state.stock_data:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.stock_data)
            
            # Clean and format data
            numeric_columns = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'free_cash_flow', 
                             'peg_ratio', 'ev_sales_ratio', 'roic', 'wacc']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Create summary dashboard BEFORE renaming columns
            create_summary_dashboard(df)
            
            # Create display DataFrame
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
                'wacc': 'WACC (%)'
            })
            
            # Format numeric columns
            display_df['Market Cap (M)'] = display_df['Market Cap (M)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) and x != 0 else "N/A")
            display_df['FCF (M)'] = display_df['FCF (M)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) and x != 0 else "N/A")
            display_df['Price ($)'] = display_df['Price ($)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and x != 0 else "N/A")
            
            # Apply filters
            filtered_df = display_df[
                (pd.to_numeric(display_df['PE Ratio'], errors='coerce').between(pe_range[0], pe_range[1], inclusive='both') | 
                 pd.to_numeric(display_df['PE Ratio'], errors='coerce').isna()) &
                (pd.to_numeric(display_df['P/B Ratio'], errors='coerce').between(pb_range[0], pb_range[1], inclusive='both') | 
                 pd.to_numeric(display_df['P/B Ratio'], errors='coerce').isna()) &
                (pd.to_numeric(display_df['ROIC (%)'], errors='coerce') >= roic_min)
            ]
            
            # Display filtered results
            st.markdown("### 📋 Screening Results")
            st.markdown(f"**{len(filtered_df)}** companies match your criteria out of **{len(display_df)}** analyzed")
            
            if not filtered_df.empty:
                # Sort options
                col1, col2 = st.columns(2)
                with col1:
                    sort_column = st.selectbox("Sort by:", 
                                             ['PE Ratio', 'P/B Ratio', 'ROIC (%)', 'PEG Ratio', 'Market Cap (M)'])
                with col2:
                    sort_order = st.selectbox("Order:", ['Ascending', 'Descending'])
                
                # Sort DataFrame
                sort_ascending = sort_order == 'Ascending'
                if sort_column in ['Market Cap (M)', 'FCF (M)']:
                    # Handle formatted strings
                    temp_col = pd.to_numeric(filtered_df[sort_column].str.replace(',', '').str.replace('N/A', '0'), errors='coerce')
                    filtered_df = filtered_df.iloc[temp_col.argsort()]
                    if not sort_ascending:
                        filtered_df = filtered_df.iloc[::-1]
                else:
                    filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')
                
                # Display the table with custom styling
                st.markdown('<div class="data-table">', unsafe_allow_html=True)
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Export options
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"undervalued_compounders_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Investment insights
                st.markdown("### 💡 Investment Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Best Value Opportunities:**")
                    low_pe_df = filtered_df[pd.to_numeric(filtered_df['PE Ratio'], errors='coerce') < 15].head(3)
                    if not low_pe_df.empty:
                        for _, row in low_pe_df.iterrows():
                            st.markdown(f"• **{row['Symbol']}** - P/E: {row['PE Ratio']}")
                    else:
                        st.markdown("No stocks with P/E < 15 found")
                
                with col2:
                    st.markdown("**🚀 High ROIC Champions:**")
                    high_roic_df = filtered_df[pd.to_numeric(filtered_df['ROIC (%)'], errors='coerce') > 15].head(3)
                    if not high_roic_df.empty:
                        for _, row in high_roic_df.iterrows():
                            st.markdown(f"• **{row['Symbol']}** - ROIC: {row['ROIC (%)']}%")
                    else:
                        st.markdown("No stocks with ROIC > 15% found")
                
            else:
                st.warning("No companies match your current filter criteria. Try adjusting the filters.")
        
        else:
            st.info("👆 Click 'Fetch Latest Data' to start screening stocks!")
    
    else:
        st.info(f"🚧 {selected_tab} screener is coming soon! Stay tuned for more screening strategies.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            📊 Stock Screener Pro | Powered by Finnhub API | 
            <em>Data is for informational purposes only. Not financial advice.</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
