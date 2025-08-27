import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class StockScreener:
    def __init__(self):
        self.data = None
        self.filtered_data = None
    
    @st.cache_data(ttl=3600)
    def fetch_stock_data(_self, symbols):
        """Fetch stock data for given symbols"""
        stock_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f'Fetching data for {symbol}... ({i+1}/{len(symbols)})')
                progress_bar.progress((i + 1) / len(symbols))
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1y")
                
                if len(hist) == 0:
                    continue
                
                # Calculate financial metrics
                current_price = hist['Close'].iloc[-1] if len(hist) > 0 else info.get('currentPrice', 0)
                
                stock_info = {
                    'Symbol': symbol,
                    'Company Name': info.get('longName', symbol),
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown'),
                    'Current Price': current_price,
                    'Market Cap': info.get('marketCap', 0),
                    'PE Ratio': info.get('trailingPE', np.nan),
                    'PEG Ratio': info.get('pegRatio', np.nan),
                    'Price to Book': info.get('priceToBook', np.nan),
                    'Debt to Equity': info.get('debtToEquity', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'ROIC': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
                    'Revenue Growth': info.get('revenueGrowth', np.nan),
                    'Earnings Growth': info.get('earningsGrowth', np.nan),
                    'Gross Margins': info.get('grossMargins', np.nan),
                    'Operating Margins': info.get('operatingMargins', np.nan),
                    'Profit Margins': info.get('profitMargins', np.nan),
                    'Current Ratio': info.get('currentRatio', np.nan),
                    'Quick Ratio': info.get('quickRatio', np.nan),
                    'Beta': info.get('beta', np.nan),
                    'Dividend Yield': info.get('dividendYield', 0),
                    '52 Week High': info.get('fiftyTwoWeekHigh', np.nan),
                    '52 Week Low': info.get('fiftyTwoWeekLow', np.nan),
                }
                
                # Calculate additional metrics
                if len(hist) >= 252:  # One year of data
                    stock_info['1Y Return'] = ((current_price - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252]) * 100
                else:
                    stock_info['1Y Return'] = np.nan
                
                if len(hist) >= 21:  # One month of data
                    stock_info['1M Return'] = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100
                else:
                    stock_info['1M Return'] = np.nan
                
                # Calculate volatility
                if len(hist) > 1:
                    returns = hist['Close'].pct_change().dropna()
                    stock_info['Volatility'] = returns.std() * np.sqrt(252) * 100
                else:
                    stock_info['Volatility'] = np.nan
                
                stock_data.append(stock_info)
                
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(stock_data)
    
    def get_sp500_symbols(self):
        """Get S&P 500 symbols from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            return sp500_table['Symbol'].tolist()
        except:
            # Fallback list of popular stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
                    'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'ADBE', 'NFLX', 'CRM', 'TMO']

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Stock Screener Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize screener
    screener = StockScreener()
    
    # Sidebar for controls
    st.sidebar.title("🔍 Screening Controls")
    
    # Data source selection
    st.sidebar.subheader("📈 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["S&P 500 Stocks", "Custom Stock List", "Upload CSV"]
    )
    
    symbols = []
    
    if data_source == "S&P 500 Stocks":
        symbols = screener.get_sp500_symbols()
        st.sidebar.info(f"Loading {len(symbols)} S&P 500 stocks")
        
    elif data_source == "Custom Stock List":
        custom_symbols = st.sidebar.text_area(
            "Enter stock symbols (one per line):",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
            height=100
        )
        symbols = [s.strip().upper() for s in custom_symbols.split('\n') if s.strip()]
        
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.sidebar.write("CSV Preview:")
            st.sidebar.dataframe(df_upload.head())
            
            if 'Symbol' in df_upload.columns:
                symbols = df_upload['Symbol'].tolist()
            else:
                st.sidebar.error("CSV must contain a 'Symbol' column")
    
    # Fetch data button
    if st.sidebar.button("🚀 Fetch Stock Data", type="primary"):
        if symbols:
            with st.spinner("Fetching stock data... This may take a few minutes."):
                screener.data = screener.fetch_stock_data(symbols[:50])  # Limit to 50 for performance
            st.success(f"Successfully loaded data for {len(screener.data)} stocks!")
        else:
            st.error("Please provide stock symbols first!")
    
    # Display data if available
    if screener.data is not None and not screener.data.empty:
        
        # Screening Filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Screening Filters")
        
        # Market Cap Filter
        market_cap_filter = st.sidebar.selectbox(
            "Market Cap Filter:",
            ["All", "Large Cap (>10B)", "Mid Cap (2B-10B)", "Small Cap (<2B)"]
        )
        
        # Financial Filters
        st.sidebar.markdown("**📊 Financial Metrics:**")
        
        pe_range = st.sidebar.slider(
            "P/E Ratio Range:",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 30.0),
            step=0.5
        )
        
        peg_range = st.sidebar.slider(
            "PEG Ratio Range:",
            min_value=0.0,
            max_value=5.0,
            value=(0.0, 2.0),
            step=0.1
        )
        
        roic_min = st.sidebar.number_input(
            "Minimum ROIC (%):",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )
        
        roe_min = st.sidebar.number_input(
            "Minimum ROE (%):",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0
        )
        
        debt_equity_max = st.sidebar.number_input(
            "Maximum Debt-to-Equity:",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        # Growth Filters
        st.sidebar.markdown("**📈 Growth Metrics:**")
        revenue_growth_min = st.sidebar.number_input(
            "Minimum Revenue Growth (%):",
            min_value=-100.0,
            max_value=200.0,
            value=5.0,
            step=1.0
        )
        
        earnings_growth_min = st.sidebar.number_input(
            "Minimum Earnings Growth (%):",
            min_value=-100.0,
            max_value=200.0,
            value=10.0,
            step=1.0
        )
        
        # Sector Filter
        sectors = ["All"] + sorted(screener.data['Sector'].unique().tolist())
        sector_filter = st.sidebar.selectbox("Sector Filter:", sectors)
        
        # Apply Filters
        if st.sidebar.button("🔍 Apply Filters"):
            filtered_df = screener.data.copy()
            
            # Market Cap Filter
            if market_cap_filter == "Large Cap (>10B)":
                filtered_df = filtered_df[filtered_df['Market Cap'] > 10e9]
            elif market_cap_filter == "Mid Cap (2B-10B)":
                filtered_df = filtered_df[(filtered_df['Market Cap'] >= 2e9) & (filtered_df['Market Cap'] <= 10e9)]
            elif market_cap_filter == "Small Cap (<2B)":
                filtered_df = filtered_df[filtered_df['Market Cap'] < 2e9]
            
            # Financial Filters
            filtered_df = filtered_df[
                (filtered_df['PE Ratio'].between(pe_range[0], pe_range[1], na_value=False)) &
                (filtered_df['PEG Ratio'].between(peg_range[0], peg_range[1], na_value=False)) &
                (filtered_df['ROIC'] >= roic_min) &
                (filtered_df['ROE'] >= roe_min) &
                (filtered_df['Debt to Equity'] <= debt_equity_max)
            ]
            
            # Growth Filters
            filtered_df = filtered_df[
                (filtered_df['Revenue Growth'] >= revenue_growth_min/100) &
                (filtered_df['Earnings Growth'] >= earnings_growth_min/100)
            ]
            
            # Sector Filter
            if sector_filter != "All":
                filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
            
            screener.filtered_data = filtered_df
            st.success(f"Filters applied! {len(filtered_df)} stocks match your criteria.")
        
        # Display Results
        df_to_display = screener.filtered_data if screener.filtered_data is not None else screener.data
        
        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(df_to_display))
        with col2:
            avg_pe = df_to_display['PE Ratio'].mean()
            st.metric("Avg P/E Ratio", f"{avg_pe:.1f}" if not np.isnan(avg_pe) else "N/A")
        with col3:
            avg_roe = df_to_display['ROE'].mean()
            st.metric("Avg ROE", f"{avg_roe:.1f}%" if not np.isnan(avg_roe) else "N/A")
        with col4:
            avg_return = df_to_display['1Y Return'].mean()
            st.metric("Avg 1Y Return", f"{avg_return:.1f}%" if not np.isnan(avg_return) else "N/A")
        
        # Charts
        st.subheader("📈 Visual Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # P/E vs PEG Ratio Scatter Plot
            fig_scatter = px.scatter(
                df_to_display,
                x='PE Ratio',
                y='PEG Ratio',
                color='Sector',
                hover_name='Symbol',
                title='P/E Ratio vs PEG Ratio',
                labels={'PE Ratio': 'P/E Ratio', 'PEG Ratio': 'PEG Ratio'}
            )
            fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="red", 
                                annotation_text="PEG = 1.0 (Fair Value)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with chart_col2:
            # Market Cap Distribution
            fig_hist = px.histogram(
                df_to_display,
                x='Market Cap',
                nbins=20,
                title='Market Cap Distribution',
                labels={'Market Cap': 'Market Cap ($)', 'count': 'Number of Stocks'}
            )
            fig_hist.update_layout(xaxis_type="log")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Sector Performance
        if len(df_to_display) > 0:
            sector_performance = df_to_display.groupby('Sector').agg({
                'Symbol': 'count',
                '1Y Return': 'mean',
                'ROE': 'mean',
                'PE Ratio': 'mean'
            }).round(2)
            sector_performance.columns = ['Count', 'Avg 1Y Return (%)', 'Avg ROE (%)', 'Avg P/E']
            
            fig_sector = px.bar(
                sector_performance.reset_index(),
                x='Sector',
                y='Avg 1Y Return (%)',
                title='Average 1-Year Return by Sector',
                color='Avg 1Y Return (%)',
                color_continuous_scale='RdYlGn'
            )
            fig_sector.update_xaxes(tickangle=45)
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Data Table
        st.subheader("📋 Stock Data")
        
        # Column selection
        all_columns = df_to_display.columns.tolist()
        default_columns = ['Symbol', 'Company Name', 'Sector', 'Current Price', 'Market Cap', 
                          'PE Ratio', 'PEG Ratio', 'ROIC', 'ROE', '1Y Return']
        available_defaults = [col for col in default_columns if col in all_columns]
        
        selected_columns = st.multiselect(
            "Select columns to display:",
            all_columns,
            default=available_defaults
        )
        
        if selected_columns:
            display_df = df_to_display[selected_columns].copy()
            
            # Format numerical columns
            for col in display_df.columns:
                if col in ['Current Price', 'Market Cap']:
                    if col == 'Market Cap':
                        display_df[col] = display_df[col].apply(lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else "N/A")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                elif col in ['PE Ratio', 'PEG Ratio', 'ROIC', 'ROE', '1Y Return', '1M Return', 'Volatility']:
                    if col in ['ROIC', 'ROE', '1Y Return', '1M Return', 'Volatility']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_to_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"stock_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Filter Summary
        if screener.filtered_data is not None:
            st.subheader("🎯 Filter Summary")
            st.markdown(f"""
            **Applied Filters:**
            - Market Cap: {market_cap_filter}
            - P/E Ratio: {pe_range[0]} - {pe_range[1]}
            - PEG Ratio: {peg_range[0]} - {peg_range[1]}
            - Minimum ROIC: {roic_min}%
            - Minimum ROE: {roe_min}%
            - Maximum Debt-to-Equity: {debt_equity_max}
            - Minimum Revenue Growth: {revenue_growth_min}%
            - Minimum Earnings Growth: {earnings_growth_min}%
            - Sector: {sector_filter}
            
            **Results:** {len(screener.filtered_data)} out of {len(screener.data)} stocks match your criteria.
            """)
    
    else:
        st.info("👆 Please select a data source and click 'Fetch Stock Data' to begin screening.")
        
        # Sample data preview
        st.subheader("🔍 What this screener can do:")
        st.markdown("""
        - **📊 Comprehensive Analysis**: Analyze stocks with 20+ financial metrics
        - **🎯 Advanced Filtering**: Filter by market cap, P/E, PEG, ROIC, ROE, growth rates, and more
        - **📈 Visual Insights**: Interactive charts and sector analysis
        - **📋 Flexible Data Sources**: S&P 500, custom lists, or CSV uploads
        - **💾 Export Results**: Download filtered results as CSV
        
        **Key Metrics Analyzed:**
        - Valuation: P/E Ratio, PEG Ratio, Price-to-Book
        - Profitability: ROE, ROIC, Profit Margins
        - Growth: Revenue Growth, Earnings Growth
        - Financial Health: Debt-to-Equity, Current Ratio
        - Performance: 1Y Return, Volatility, Beta
        """)

if __name__ == "__main__":
    main()
