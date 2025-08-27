import streamlit as st
import sys

# Debug information
st.write("🔍 Debug Info:")
st.write(f"Python version: {sys.version}")
st.write("Checking imports...")

# Check for required packages with detailed error reporting
try:
    import pandas as pd
    st.success("✅ pandas imported")
except ImportError as e:
    st.error(f"❌ pandas failed: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported")
except ImportError as e:
    st.error(f"❌ numpy failed: {e}")

try:
    import yfinance as yf
    st.success("✅ yfinance imported")
except ImportError as e:
    st.error(f"❌ yfinance failed: {e}")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    st.success("✅ plotly imported")
except ImportError as e:
    st.error(f"❌ plotly failed: {e}")

try:
    import requests
    st.success("✅ requests imported")
except ImportError as e:
    st.error(f"❌ requests failed: {e}")

try:
    from datetime import datetime, timedelta
    import time
    import warnings
    warnings.filterwarnings('ignore')
    st.success("✅ All standard libraries imported")
except ImportError as e:
    st.error(f"❌ Standard libraries failed: {e}")

st.write("---")
st.write("If all imports are successful, the error is elsewhere. Continuing with app...")

# Page configuration
st.set_page_config(
    page_title="Value Investment Screener",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force cache clear for deployment issues
if 'deployment_id' not in st.session_state:
    st.session_state.deployment_id = True

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .metric-good { background: linear-gradient(90deg, #28a745, #20c997); color: white; padding: 10px; border-radius: 8px; }
    .metric-warning { background: linear-gradient(90deg, #ffc107, #fd7e14); color: white; padding: 10px; border-radius: 8px; }
    .metric-danger { background: linear-gradient(90deg, #dc3545, #e83e8c); color: white; padding: 10px; border-radius: 8px; }
    .investment-tip { background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; border-radius: 5px; margin: 10px 0; }
    .tooltip { position: relative; display: inline-block; cursor: help; color: #007bff; }
</style>
""", unsafe_allow_html=True)

# Metric tooltips for layman understanding
METRIC_TOOLTIPS = {
    "PE Ratio": "Price-to-Earnings: How expensive is the stock? Lower = Cheaper. Good value: 5-15",
    "PEG Ratio": "PE vs Growth: Is the price fair for growth? Under 1.0 = Undervalued gem!",
    "Price to Book": "Price vs Book Value: Market price vs company's worth. Under 1.5 = Good deal",
    "ROIC": "Return on Capital: How efficiently company uses money. Above 15% = Excellent management",
    "ROE": "Return on Equity: Profit from shareholders' money. Above 15% = Great returns",
    "Debt to Equity": "Company's debt burden. Under 0.5 = Safe, 0.5-1 = Manageable, Above 1 = Risky",
    "Current Ratio": "Can pay short-term bills? Above 1.5 = Financially healthy",
    "Revenue Growth": "Sales increasing? Above 10% = Growing business",
    "Earnings Growth": "Profits increasing? Above 15% = Excellent growth",
    "Dividend Yield": "Annual dividend as % of price. 2-6% = Good income stock",
    "Market Cap": "Company size: Small (<2B), Mid (2-10B), Large (10-50B), Mega (>50B)"
}

def show_tooltip(metric_name):
    """Display tooltip for metric"""
    if metric_name in METRIC_TOOLTIPS:
        return f" ℹ️ {METRIC_TOOLTIPS[metric_name]}"
    return ""

class ValueInvestmentScreener:
    def __init__(self):
        self.data = None
        self.investment_opportunities = None
    
    def get_quality_stock_symbols(self):
        """Get symbols of quality large and mid-cap companies"""
        # Focus on established companies more likely to have complete data
        quality_symbols = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'INTC',
            # Large Cap Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BRK-B', 'V', 'MA',
            # Large Cap Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD',
            # Large Cap Consumer
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
            # Large Cap Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR',
            # Mid Cap Growth
            'ROKU', 'SQ', 'TWLO', 'OKTA', 'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'SNOW'
        ]
        return quality_symbols
    
    @st.cache_data(ttl=3600)
    def fetch_comprehensive_stock_data(_self, symbols):
        """Fetch comprehensive stock data with better error handling"""
        stock_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f'📊 Analyzing {symbol}... ({i+1}/{len(symbols)})')
                progress_bar.progress((i + 1) / len(symbols))
                
                ticker = yf.Ticker(symbol)
                
                # Get multiple data sources
                info = ticker.info
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                hist = ticker.history(period="2y")  # 2 years for better calculations
                
                if len(hist) == 0 or not info:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Calculate comprehensive metrics with fallbacks
                stock_info = {
                    'Symbol': symbol,
                    'Company Name': info.get('longName', info.get('shortName', symbol)),
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown'),
                    'Current Price': round(current_price, 2),
                }
                
                # Market Cap and Size Classification
                market_cap = info.get('marketCap', 0)
                stock_info['Market Cap'] = market_cap
                
                if market_cap > 50e9:
                    stock_info['Size Category'] = 'Mega Cap'
                elif market_cap > 10e9:
                    stock_info['Size Category'] = 'Large Cap'
                elif market_cap > 2e9:
                    stock_info['Size Category'] = 'Mid Cap'
                else:
                    stock_info['Size Category'] = 'Small Cap'
                
                # Valuation Metrics (Key for Value Investing)
                stock_info['PE Ratio'] = info.get('trailingPE', info.get('forwardPE', np.nan))
                stock_info['PEG Ratio'] = info.get('pegRatio', np.nan)
                stock_info['Price to Book'] = info.get('priceToBook', np.nan)
                stock_info['EV/Revenue'] = info.get('enterpriseToRevenue', np.nan)
                
                # Profitability Metrics (Quality indicators)
                stock_info['ROE'] = info.get('returnOnEquity', np.nan)
                if stock_info['ROE'] and not np.isnan(stock_info['ROE']):
                    stock_info['ROE'] = stock_info['ROE'] * 100
                
                stock_info['ROIC'] = info.get('returnOnAssets', np.nan)
                if stock_info['ROIC'] and not np.isnan(stock_info['ROIC']):
                    stock_info['ROIC'] = stock_info['ROIC'] * 100
                
                stock_info['Gross Margins'] = info.get('grossMargins', np.nan)
                if stock_info['Gross Margins'] and not np.isnan(stock_info['Gross Margins']):
                    stock_info['Gross Margins'] = stock_info['Gross Margins'] * 100
                
                stock_info['Operating Margins'] = info.get('operatingMargins', np.nan)
                if stock_info['Operating Margins'] and not np.isnan(stock_info['Operating Margins']):
                    stock_info['Operating Margins'] = stock_info['Operating Margins'] * 100
                
                stock_info['Profit Margins'] = info.get('profitMargins', np.nan)
                if stock_info['Profit Margins'] and not np.isnan(stock_info['Profit Margins']):
                    stock_info['Profit Margins'] = stock_info['Profit Margins'] * 100
                
                # Growth Metrics
                stock_info['Revenue Growth'] = info.get('revenueGrowth', np.nan)
                if stock_info['Revenue Growth'] and not np.isnan(stock_info['Revenue Growth']):
                    stock_info['Revenue Growth'] = stock_info['Revenue Growth'] * 100
                
                stock_info['Earnings Growth'] = info.get('earningsGrowth', info.get('earningsQuarterlyGrowth', np.nan))
                if stock_info['Earnings Growth'] and not np.isnan(stock_info['Earnings Growth']):
                    stock_info['Earnings Growth'] = stock_info['Earnings Growth'] * 100
                
                # Financial Health
                stock_info['Debt to Equity'] = info.get('debtToEquity', np.nan)
                if stock_info['Debt to Equity'] and not np.isnan(stock_info['Debt to Equity']):
                    stock_info['Debt to Equity'] = stock_info['Debt to Equity'] / 100
                
                stock_info['Current Ratio'] = info.get('currentRatio', np.nan)
                stock_info['Quick Ratio'] = info.get('quickRatio', np.nan)
                
                # Performance and Risk
                if len(hist) >= 252:  # 1 year
                    stock_info['1Y Return'] = ((current_price - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252]) * 100
                else:
                    stock_info['1Y Return'] = np.nan
                
                if len(hist) >= 63:  # 3 months
                    stock_info['3M Return'] = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100
                else:
                    stock_info['3M Return'] = np.nan
                
                # Dividend Information
                stock_info['Dividend Yield'] = info.get('dividendYield', 0)
                if stock_info['Dividend Yield']:
                    stock_info['Dividend Yield'] = stock_info['Dividend Yield'] * 100
                
                # Beta and Volatility
                stock_info['Beta'] = info.get('beta', np.nan)
                
                if len(hist) > 20:
                    returns = hist['Close'].pct_change().dropna()
                    stock_info['Volatility'] = returns.std() * np.sqrt(252) * 100
                else:
                    stock_info['Volatility'] = np.nan
                
                # 52-week range
                stock_info['52W High'] = info.get('fiftyTwoWeekHigh', np.nan)
                stock_info['52W Low'] = info.get('fiftyTwoWeekLow', np.nan)
                
                # Calculate position in 52-week range
                if stock_info['52W High'] and stock_info['52W Low']:
                    stock_info['52W Position'] = ((current_price - stock_info['52W Low']) / 
                                                (stock_info['52W High'] - stock_info['52W Low'])) * 100
                else:
                    stock_info['52W Position'] = np.nan
                
                stock_data.append(stock_info)
                
            except Exception as e:
                st.warning(f"⚠️ Could not fetch complete data for {symbol}: {str(e)}")
                continue
            
            time.sleep(0.2)  # Respectful delay
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(stock_data)
    
    def identify_investment_opportunities(self, df):
        """Apply value investing filters to identify opportunities"""
        
        if df.empty:
            return df
        
        # Value Investing Criteria (Warren Buffett / Benjamin Graham style)
        opportunities = df.copy()
        
        # Filter 1: Focus on Mid, Large, and Mega Cap (avoid small cap volatility)
        opportunities = opportunities[opportunities['Size Category'].isin(['Mid Cap', 'Large Cap', 'Mega Cap'])]
        
        # Filter 2: Undervaluation Metrics
        # PEG < 1.0 (Growth at a reasonable price)
        opportunities = opportunities[
            (opportunities['PEG Ratio'] > 0) & 
            (opportunities['PEG Ratio'] < 1.0)
        ]
        
        # Filter 3: Quality Metrics (Profitable companies)
        opportunities = opportunities[
            (opportunities['ROE'] > 15) |  # Strong return on equity
            (opportunities['ROIC'] > 12)   # Efficient capital usage
        ]
        
        # Filter 4: Financial Health
        opportunities = opportunities[
            (opportunities['Debt to Equity'] < 1.0) |  # Manageable debt
            (opportunities['Current Ratio'] > 1.2)     # Can pay bills
        ]
        
        # Filter 5: Growth Potential
        opportunities = opportunities[
            (opportunities['Revenue Growth'] > 5) |    # Growing business
            (opportunities['Earnings Growth'] > 10)    # Growing profits
        ]
        
        # Calculate Investment Score (0-100)
        def calculate_investment_score(row):
            score = 50  # Base score
            
            # Valuation bonus (cheaper is better)
            if pd.notna(row['PEG Ratio']):
                if row['PEG Ratio'] < 0.5:
                    score += 20
                elif row['PEG Ratio'] < 0.8:
                    score += 15
                elif row['PEG Ratio'] < 1.0:
                    score += 10
            
            # Quality bonus
            if pd.notna(row['ROE']) and row['ROE'] > 20:
                score += 10
            if pd.notna(row['ROIC']) and row['ROIC'] > 15:
                score += 10
            
            # Growth bonus
            if pd.notna(row['Revenue Growth']) and row['Revenue Growth'] > 15:
                score += 8
            if pd.notna(row['Earnings Growth']) and row['Earnings Growth'] > 20:
                score += 8
            
            # Financial health bonus
            if pd.notna(row['Debt to Equity']) and row['Debt to Equity'] < 0.3:
                score += 5
            if pd.notna(row['Current Ratio']) and row['Current Ratio'] > 2.0:
                score += 5
            
            # Recent performance penalty/bonus
            if pd.notna(row['1Y Return']):
                if row['1Y Return'] < -20:  # Beaten down stock (opportunity)
                    score += 5
                elif row['1Y Return'] > 50:  # Overheated (risk)
                    score -= 5
            
            return min(100, max(0, score))
        
        opportunities['Investment Score'] = opportunities.apply(calculate_investment_score, axis=1)
        
        # Sort by investment score
        opportunities = opportunities.sort_values('Investment Score', ascending=False)
        
        return opportunities

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Value Investment Screener</h1>', unsafe_allow_html=True)
    st.markdown("### Find Undervalued Growth Stocks for Long-term Investment")
    
    # Investment Philosophy
    st.markdown("""
    <div class="investment-tip">
    <strong>🎯 Investment Philosophy:</strong> This screener finds undervalued companies with strong fundamentals, 
    manageable debt, and growth potential - perfect for long-term wealth building.
    </div>
    """, unsafe_allow_html=True)
    
    screener = ValueInvestmentScreener()
    
    # Sidebar Controls
    st.sidebar.title("🔍 Screening Parameters")
    
    # Custom criteria
    st.sidebar.subheader("💰 Value Criteria")
    
    max_peg = st.sidebar.slider(
        f"Maximum PEG Ratio{show_tooltip('PEG Ratio')}", 
        0.1, 2.0, 1.0, 0.1,
        help=METRIC_TOOLTIPS['PEG Ratio']
    )
    
    max_pe = st.sidebar.slider(
        f"Maximum PE Ratio{show_tooltip('PE Ratio')}", 
        5.0, 30.0, 20.0, 1.0,
        help=METRIC_TOOLTIPS['PE Ratio']
    )
    
    max_pb = st.sidebar.slider(
        f"Maximum Price-to-Book{show_tooltip('Price to Book')}", 
        0.5, 3.0, 2.0, 0.1,
        help=METRIC_TOOLTIPS['Price to Book']
    )
    
    st.sidebar.subheader("📈 Quality & Growth")
    
    min_roe = st.sidebar.slider(
        f"Minimum ROE (%){show_tooltip('ROE')}", 
        0.0, 30.0, 15.0, 1.0,
        help=METRIC_TOOLTIPS['ROE']
    )
    
    min_revenue_growth = st.sidebar.slider(
        f"Minimum Revenue Growth (%){show_tooltip('Revenue Growth')}", 
        -10.0, 30.0, 5.0, 1.0,
        help=METRIC_TOOLTIPS['Revenue Growth']
    )
    
    st.sidebar.subheader("🛡️ Financial Safety")
    
    max_debt_equity = st.sidebar.slider(
        f"Maximum Debt-to-Equity{show_tooltip('Debt to Equity')}", 
        0.0, 2.0, 0.8, 0.1,
        help=METRIC_TOOLTIPS['Debt to Equity']
    )
    
    company_sizes = st.sidebar.multiselect(
        f"Company Size{show_tooltip('Market Cap')}", 
        ['Mid Cap', 'Large Cap', 'Mega Cap'], 
        default=['Mid Cap', 'Large Cap', 'Mega Cap'],
        help=METRIC_TOOLTIPS['Market Cap']
    )
    
    # Scan button
    if st.sidebar.button("🚀 Find Investment Opportunities", type="primary"):
        with st.spinner("🔍 Scanning the market for value opportunities..."):
            symbols = screener.get_quality_stock_symbols()
            screener.data = screener.fetch_comprehensive_stock_data(symbols)
            
            if not screener.data.empty:
                # Apply custom filters
                filtered_data = screener.data.copy()
                
                # Apply user-defined filters
                if company_sizes:
                    filtered_data = filtered_data[filtered_data['Size Category'].isin(company_sizes)]
                
                filtered_data = filtered_data[
                    (filtered_data['PEG Ratio'] <= max_peg) &
                    (filtered_data['PE Ratio'] <= max_pe) &
                    (filtered_data['Price to Book'] <= max_pb) &
                    (filtered_data['ROE'] >= min_roe) &
                    (filtered_data['Revenue Growth'] >= min_revenue_growth) &
                    (filtered_data['Debt to Equity'] <= max_debt_equity)
                ]
                
                screener.investment_opportunities = screener.identify_investment_opportunities(filtered_data)
                st.success(f"✅ Found {len(screener.investment_opportunities)} investment opportunities!")
            else:
                st.error("❌ No data could be fetched. Please try again later.")
    
    # Display Results
    if screener.investment_opportunities is not None and not screener.investment_opportunities.empty:
        
        # Summary Dashboard
        st.subheader("📊 Investment Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        opportunities = screener.investment_opportunities
        
        with col1:
            st.metric("🎯 Opportunities Found", len(opportunities))
        
        with col2:
            avg_score = opportunities['Investment Score'].mean()
            st.metric("📈 Avg Investment Score", f"{avg_score:.1f}/100")
        
        with col3:
            avg_peg = opportunities['PEG Ratio'].mean()
            st.metric("💎 Avg PEG Ratio", f"{avg_peg:.2f}")
        
        with col4:
            avg_roe = opportunities['ROE'].mean()
            st.metric("⚡ Avg ROE", f"{avg_roe:.1f}%")
        
        # Top Investment Picks
        st.subheader("🏆 Top Investment Opportunities")
        
        # Display top 10 opportunities
        top_picks = opportunities.head(10)
        
        for i, (idx, stock) in enumerate(top_picks.iterrows()):
            with st.expander(f"#{i+1} {stock['Symbol']} - {stock['Company Name']} (Score: {stock['Investment Score']:.0f}/100)"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Valuation**")
                    st.write(f"PE Ratio: {stock['PE Ratio']:.2f}" if pd.notna(stock['PE Ratio']) else "PE Ratio: N/A")
                    st.write(f"PEG Ratio: {stock['PEG Ratio']:.2f}" if pd.notna(stock['PEG Ratio']) else "PEG Ratio: N/A")
                    st.write(f"Price/Book: {stock['Price to Book']:.2f}" if pd.notna(stock['Price to Book']) else "Price/Book: N/A")
                
                with col2:
                    st.markdown("**🚀 Growth & Quality**")
                    st.write(f"ROE: {stock['ROE']:.1f}%" if pd.notna(stock['ROE']) else "ROE: N/A")
                    st.write(f"Revenue Growth: {stock['Revenue Growth']:.1f}%" if pd.notna(stock['Revenue Growth']) else "Revenue Growth: N/A")
                    st.write(f"Earnings Growth: {stock['Earnings Growth']:.1f}%" if pd.notna(stock['Earnings Growth']) else "Earnings Growth: N/A")
                
                with col3:
                    st.markdown("**🛡️ Safety & Performance**")
                    st.write(f"Debt/Equity: {stock['Debt to Equity']:.2f}" if pd.notna(stock['Debt to Equity']) else "Debt/Equity: N/A")
                    st.write(f"Current Ratio: {stock['Current Ratio']:.2f}" if pd.notna(stock['Current Ratio']) else "Current Ratio: N/A")
                    st.write(f"1Y Return: {stock['1Y Return']:.1f}%" if pd.notna(stock['1Y Return']) else "1Y Return: N/A")
                
                # Investment rationale
                rationale = []
                if pd.notna(stock['PEG Ratio']) and stock['PEG Ratio'] < 0.8:
                    rationale.append("🎯 Excellent value with PEG < 0.8")
                if pd.notna(stock['ROE']) and stock['ROE'] > 20:
                    rationale.append("⚡ Strong profitability (ROE > 20%)")
                if pd.notna(stock['Debt to Equity']) and stock['Debt to Equity'] < 0.3:
                    rationale.append("🛡️ Very low debt burden")
                if pd.notna(stock['Revenue Growth']) and stock['Revenue Growth'] > 15:
                    rationale.append("📈 Strong revenue growth")
                
                if rationale:
                    st.markdown("**💡 Why this is a good opportunity:**")
                    for reason in rationale:
                        st.write(f"• {reason}")
        
        # Data Table
        st.subheader("📋 Complete Analysis")
        
        # Key columns for investors
        display_cols = [
            'Symbol', 'Company Name', 'Size Category', 'Current Price',
            'Investment Score', 'PE Ratio', 'PEG Ratio', 'ROE', 'ROIC',
            'Revenue Growth', 'Debt to Equity', '1Y Return'
        ]
        
        available_cols = [col for col in display_cols if col in opportunities.columns]
        
        st.dataframe(
            opportunities[available_cols].round(2),
            use_container_width=True,
            height=400
        )
        
        # Download Results
        csv = opportunities.to_csv(index=False)
        st.download_button(
            label="📥 Download Investment Opportunities",
            data=csv,
            file_name=f"investment_opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Market Analysis
        if len(opportunities) > 0:
            st.subheader("📈 Market Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PEG vs Score scatter
                fig_scatter = px.scatter(
                    opportunities, 
                    x='PEG Ratio', 
                    y='Investment Score',
                    color='Size Category',
                    hover_name='Symbol',
                    title='PEG Ratio vs Investment Score',
                    labels={'PEG Ratio': 'PEG Ratio (Lower = Better Value)', 'Investment Score': 'Investment Score'}
                )
                fig_scatter.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Fair Value (PEG=1)")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Sector distribution
                if 'Sector' in opportunities.columns:
                    sector_counts = opportunities['Sector'].value_counts()
                    fig_pie = px.pie(
                        values=sector_counts.values,
                        names=sector_counts.index,
                        title='Investment Opportunities by Sector'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
    
    elif screener.investment_opportunities is not None and screener.investment_opportunities.empty:
        st.warning("🤔 No opportunities found with current criteria. Try relaxing some filters.")
        st.info("💡 **Tips to find opportunities:**\n- Increase maximum PEG ratio\n- Lower minimum ROE\n- Include smaller companies")
    
    else:
        # Welcome screen
        st.info("👆 Click 'Find Investment Opportunities' to start screening!")
        
        st.markdown("""
        ### 🎯 What This Screener Finds:
        
        **Value Opportunities:**
        - Stocks trading below fair value (PEG < 1.0)
        - Strong companies at reasonable prices
        - Hidden gems before market discovers them
        
        **Quality Companies:**
        - Profitable businesses (High ROE/ROIC)
        - Growing revenue and earnings
        - Strong balance sheets
        
        **Investment Safety:**
        - Focus on Mid-Large cap stability
        - Low debt companies
        - Financially healthy businesses
        
        ### 📚 Metric Guide:
        """)
        
        with st.expander("🔍 Understanding the Metrics (Click to expand)"):
            for metric, explanation in METRIC_TOOLTIPS.items():
                st.write(f"**{metric}:** {explanation}")

if __name__ == "__main__":
    main()
