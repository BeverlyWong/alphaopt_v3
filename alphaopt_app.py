import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(
    page_title="AlphaOpt - ETF Portfolio System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #374151;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4b5563;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

ETF_CLASS_MAP = {
    'AGG': 'US Broad Bond',
    'BND': 'US Broad Bond',
    'GLD': 'Commodity - Gold',
    'HYG': 'US High Yield Bond',
    'LQD': 'US Investment Grade Corp Bond',
    'TLT': 'US Long-Term Treasury',
    'XLE': 'US Energy',
    'XLF': 'US Financials',
    'XLI': 'US Industrials',
    'XLK': 'US Technology Sector',
    'XLP': 'US Consumer Staples',
    'XLU': 'US Utilities',
    'XLV': 'US Healthcare',
    'XLY': 'US Consumer Discretionary',
    'XLB': 'US Materials',
    'XLRE': 'US Real Estate',
    'SPY': 'US Large Cap',
    'IVV': 'US Large Cap',
    'VTI': 'US Total Market',
    'QQQ': 'US Technology',
    'VUG': 'US Large Growth',
    'VTV': 'US Large Value',
    'IWM': 'US Small Cap',
    'IJR': 'US Small Cap',
    'VO': 'US Mid Cap',
    'VEA': 'International Developed',
    'IEFA': 'International Developed',
    'VWO': 'Emerging Markets',
    'EEM': 'Emerging Markets',
    'IEMG': 'Emerging Markets'
}

@st.cache_data
def load_data():
    data = {}
    data['etf_summary'] = pd.read_csv('data/ETF_Summary_Report_20251125_232444.csv')
    data['performance'] = pd.read_csv('data/Performance_Statistics_20251125_232444.csv')
    data['correlation'] = pd.read_csv('data/correlation_matrix.csv', index_col=0)
    data['corr_pairs'] = pd.read_csv('data/correlated_pairs.csv')
    data['class_risk'] = pd.read_csv('data/class_risk_summary.csv')
    with open('data/portfolio_weights_3types_final.json', 'r') as f:
        data['weights'] = json.load(f)
    data['expected_perf'] = pd.read_csv('data/expected_performance_3types_final.csv')
    return data

def render_overview(data):
    st.markdown('<h1 class="main-header">AlphaOpt</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent ETF Allocation & Risk Control System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### About This Project
    
    In today's complex global financial markets, individual investors face two core challenges:
    
    1. **Difficulty identifying overvalued ETFs** driven by market sentiment
    2. **Lack of professional tools** to build optimal portfolios matching their risk preferences
    
    **AlphaOpt** is a one-stop intelligent investment decision support system that combines 
    **real-time premium monitoring** with **Modern Portfolio Theory** to help investors make 
    informed decisions.
    """)
    
    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ETFs Analyzed", "30")
    with col2:
        st.metric("Analysis Period", "3 Years", delta="2022-2024")
    with col3:
        st.metric("Trading Days", "752")
    with col4:
        st.metric("Risk Levels", "3")
    
    st.markdown("---")
    
    st.markdown("### ETF Universe Overview")
    
    df = data['etf_summary'].copy()
    
    categories = ['All'] + sorted(df['Category'].unique().tolist())
    selected_category = st.selectbox("Filter by Category", categories)
    
    if selected_category != 'All':
        df = df[df['Category'] == selected_category]
    
    display_cols = ['ETF_Symbol', 'Fund_Name', 'Category', 'Latest_Price', 
                   'Total_Return_%', 'Annual_Volatility_%', 'Sharpe_Ratio']
    
    df_display = df[display_cols].copy()
    df_display.columns = ['Symbol', 'Fund Name', 'Category', 'Price ($)', 
                         'Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']
    
    df_display['Price ($)'] = df_display['Price ($)'].round(2)
    df_display['Total Return (%)'] = df_display['Total Return (%)'].round(2)
    df_display['Volatility (%)'] = df_display['Volatility (%)'].round(2)
    df_display['Sharpe Ratio'] = df_display['Sharpe Ratio'].round(3)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)


def render_analysis(data):
    st.markdown("## Performance Analysis")
    st.markdown("Comprehensive analysis of ETF performance metrics over the 3-year period.")
    
    st.markdown("---")
    
    perf = data['performance'].copy()
    perf.columns = ['ETF', 'Total_Return', 'Volatility', 'Sharpe_Ratio']
    
    tab1, tab2, tab3 = st.tabs(["📊 Return vs Risk", "📈 Performance Ranking", "📋 Asset Class Analysis"])
    
    with tab1:
        st.markdown("### Risk-Return Scatter Plot")
        
        fig = px.scatter(
            perf,
            x='Volatility',
            y='Total_Return',
            text='ETF',
            size=np.abs(perf['Sharpe_Ratio']) * 20 + 10,
            color='Sharpe_Ratio',
            color_continuous_scale='RdYlGn',
            labels={
                'Volatility': 'Annual Volatility (%)',
                'Total_Return': 'Total Return (%)',
                'Sharpe_Ratio': 'Sharpe Ratio'
            }
        )
        
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_layout(height=550)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=perf['Volatility'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("**Interpretation**: Upper-left quadrant (high return, low risk) represents the best performers. Bubble size indicates Sharpe Ratio magnitude.")
    
    with tab2:
        st.markdown("### Performance Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 by Total Return")
            top_return = perf.nlargest(10, 'Total_Return')[['ETF', 'Total_Return', 'Sharpe_Ratio']]
            top_return['Total_Return'] = top_return['Total_Return'].round(2).astype(str) + '%'
            top_return['Sharpe_Ratio'] = top_return['Sharpe_Ratio'].round(3)
            st.dataframe(top_return, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Top 10 by Sharpe Ratio")
            top_sharpe = perf.nlargest(10, 'Sharpe_Ratio')[['ETF', 'Sharpe_Ratio', 'Total_Return']]
            top_sharpe['Total_Return'] = top_sharpe['Total_Return'].round(2).astype(str) + '%'
            top_sharpe['Sharpe_Ratio'] = top_sharpe['Sharpe_Ratio'].round(3)
            st.dataframe(top_sharpe, use_container_width=True, hide_index=True)
        
        st.markdown("#### Return Comparison")
        perf_sorted = perf.sort_values('Total_Return', ascending=True)
        
        colors = ['#ef4444' if x < 0 else '#22c55e' for x in perf_sorted['Total_Return']]
        
        fig = go.Figure(go.Bar(
            x=perf_sorted['Total_Return'],
            y=perf_sorted['ETF'],
            orientation='h',
            marker_color=colors
        ))
        
        fig.update_layout(
            height=650,
            xaxis_title='Total Return (%)',
            yaxis_title='',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Asset Class Performance")
        
        class_risk = data['class_risk'].copy()
        class_risk.columns = ['Class', 'Ann_Return', 'Ann_Volatility', 'Sharpe', 'Max_Drawdown']
        
        class_risk['Ann_Return_Pct'] = (class_risk['Ann_Return'] * 100).round(2)
        class_risk['Ann_Volatility_Pct'] = (class_risk['Ann_Volatility'] * 100).round(2)
        class_risk['Max_Drawdown_Pct'] = (class_risk['Max_Drawdown'] * 100).round(2)
        
        fig = px.scatter(
            class_risk,
            x='Ann_Volatility_Pct',
            y='Ann_Return_Pct',
            text='Class',
            size=np.abs(class_risk['Sharpe']) * 30 + 15,
            color='Sharpe',
            color_continuous_scale='RdYlGn',
            labels={
                'Ann_Volatility_Pct': 'Annual Volatility (%)',
                'Ann_Return_Pct': 'Annual Return (%)',
                'Sharpe': 'Sharpe Ratio'
            }
        )
        
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(height=500)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detailed Statistics by Asset Class")
        display_class = class_risk[['Class', 'Ann_Return_Pct', 'Ann_Volatility_Pct', 'Sharpe', 'Max_Drawdown_Pct']].copy()
        display_class.columns = ['Asset Class', 'Annual Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        display_class = display_class.sort_values('Sharpe Ratio', ascending=False)
        st.dataframe(display_class, use_container_width=True, hide_index=True)


def render_correlation(data):
    st.markdown("## Correlation Analysis")
    st.markdown("Understanding the relationships between different ETFs for effective diversification.")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔥 Correlation Heatmap", "🔗 Notable Pairs"])
    
    with tab1:
        st.markdown("### 30 ETFs Correlation Heatmap")
        
        corr = data['correlation']
        
        text_values = np.round(corr.values, 2).astype(str)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale='YlGn',
            zmin=0,
            zmax=1,
            text=text_values,
            texttemplate='%{text}',
            textfont=dict(size=8),
            hovertemplate='%{x} - %{y}<br>Correlation: %{z:.2f}<extra></extra>',
            colorbar=dict(title=dict(text='Correlation', font=dict(size=12)))
        ))
        
        fig.update_layout(
            height=800,
            xaxis=dict(tickfont=dict(size=9), tickangle=45, side='bottom'),
            yaxis=dict(tickfont=dict(size=9), autorange='reversed'),
            margin=dict(l=80, r=60, t=40, b=120)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **How to Read This Heatmap:**
        - **Dark Green (close to 1)**: High positive correlation - assets move together
        - **Light Yellow (close to 0)**: Low correlation - assets move independently
        
        For diversification, combine assets with low correlations.
        """)
    
    with tab2:
        st.markdown("### Notable Correlation Pairs")
        
        pairs = data['corr_pairs'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Highly Correlated Pairs (>0.80)")
            st.caption("These pairs move together - holding both provides limited diversification")
            
            high_corr = pairs[pairs['Type'] == 'High'].sort_values('Correlation', ascending=False).head(15)
            high_corr['Correlation'] = high_corr['Correlation'].round(4)
            st.dataframe(high_corr[['ETF1', 'ETF2', 'Correlation']], use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Lowly Correlated Pairs (<0.20)")
            st.caption("These pairs move independently - good for diversification")
            
            low_corr = pairs[pairs['Type'] == 'Low'].sort_values('Correlation', ascending=True).head(15)
            low_corr['Correlation'] = low_corr['Correlation'].round(4)
            st.dataframe(low_corr[['ETF1', 'ETF2', 'Correlation']], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Diversification Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Bond-Stock Diversification")
            st.markdown("""
            - **AGG/BND** with **XLE**: Near-zero correlation
            - Bonds provide stability when energy stocks fluctuate
            """)
        
        with col2:
            st.markdown("##### Gold as Hedge")
            st.markdown("""
            - **GLD** shows low correlation (~0.15) with most equities
            - Effective portfolio hedge during market stress
            """)
        
        with col3:
            st.markdown("##### Treasury Diversification")
            st.markdown("""
            - **TLT** has low/negative correlation with stocks
            - Long-term treasuries move opposite to risk assets
            """)


def render_portfolio(data):
    st.markdown("## Portfolio Allocation")
    st.markdown("Optimized portfolios for three risk levels based on Modern Portfolio Theory.")
    
    st.markdown("---")
    
    weights = data['weights']
    
    st.markdown("### Portfolio Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    portfolios_info = [
        ('Conservative', col1, '#22c55e', 'Low risk tolerance, focus on capital preservation'),
        ('Balanced', col2, '#eab308', 'Moderate risk, balanced growth and stability'),
        ('Aggressive', col3, '#ef4444', 'High risk tolerance, maximize returns')
    ]
    
    for name, col, color, desc in portfolios_info:
        with col:
            port = weights[name]
            st.markdown(f"### {name}")
            st.caption(desc)
            
            st.metric("Expected Return", f"{port['expected_return']*100:.2f}%")
            st.metric("Volatility", f"{port['volatility']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{port['sharpe_ratio']:.3f}")
            st.metric("Max Drawdown", f"{port['max_drawdown']*100:.2f}%")
    
    st.markdown("---")
    
    st.markdown("### Efficient Frontier with Three Risk Levels")
    
    frontier_returns = [0.00, 0.007, 0.014, 0.021, 0.0286, 0.0357, 0.0429, 0.05, 0.0571, 0.0643, 0.0714, 0.0786, 0.0857, 0.0929, 0.10]
    frontier_vols = [0.0735, 0.0720, 0.0718, 0.0720, 0.0722, 0.0735, 0.0753, 0.0787, 0.0829, 0.0877, 0.0931, 0.0989, 0.1052, 0.1121, 0.1197]
    frontier_drawdowns = [-0.15, -0.148, -0.145, -0.140, -0.1178, -0.120, -0.118, -0.116, -0.115, -0.1153, -0.120, -0.125, -0.130, -0.133, -0.1365]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_returns,
        mode='markers',
        marker=dict(
            size=12,
            color=frontier_drawdowns,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Max Drawdown', x=1.02),
            showscale=True
        ),
        name='Efficient Frontier',
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    risk_levels = [
        ('Conservative', weights['Conservative']['volatility'], weights['Conservative']['expected_return'], 'circle', '#22c55e', 20),
        ('Balanced', weights['Balanced']['volatility'], weights['Balanced']['expected_return'], 'square', '#eab308', 20),
        ('Aggressive', weights['Aggressive']['volatility'], weights['Aggressive']['expected_return'], 'triangle-up', '#ef4444', 20)
    ]
    
    for name, vol, ret, symbol, color, size in risk_levels:
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers+text',
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=2, color='black')),
            text=[name],
            textposition='top center',
            textfont=dict(size=12, color=color),
            name=name,
            hovertemplate=f'{name}<br>Volatility: {vol:.2%}<br>Return: {ret:.2%}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier with Three Risk Levels (Based on 2021-2024 Data)',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.update_xaxes(tickformat='.0%')
    fig.update_yaxes(tickformat='.0%')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Asset Allocation by Class")
    
    def aggregate_weights_by_class(port_weights):
        class_weights = {}
        for etf, weight in port_weights.items():
            if weight > 0.001:
                asset_class = ETF_CLASS_MAP.get(etf, etf)
                class_weights[asset_class] = class_weights.get(asset_class, 0) + weight
        return class_weights
    
    col1, col2, col3 = st.columns(3)
    
    colors_map = {
        'Conservative': ['#2dd4bf', '#34d399', '#4ade80', '#86efac', '#a7f3d0', '#bbf7d0', '#d1fae5', '#ecfdf5'],
        'Balanced': ['#fbbf24', '#fcd34d', '#fde047', '#fef08a', '#a3e635', '#84cc16', '#65a30d', '#4d7c0f'],
        'Aggressive': ['#f87171', '#fb923c', '#fbbf24', '#facc15', '#a3e635', '#4ade80', '#2dd4bf', '#22d3ee']
    }
    
    for (name, col, color, _), column in zip(portfolios_info, [col1, col2, col3]):
        with column:
            port_weights = weights[name]['weights']
            class_weights = aggregate_weights_by_class(port_weights)
            
            sorted_items = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0,
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=10),
                marker=dict(colors=colors_map[name][:len(labels)]),
                insidetextorientation='horizontal'
            )])
            
            fig.update_layout(
                title=dict(text=f'{name} Asset Allocation', x=0.5, font=dict(size=14)),
                height=400,
                showlegend=False,
                margin=dict(t=60, b=60, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Portfolio Weights Comparison")
    
    col1, col2, col3 = st.columns(3)
    bar_colors = {'Conservative': '#22c55e', 'Balanced': '#eab308', 'Aggressive': '#ef4444'}
    
    for (name, _, _, _), column in zip(portfolios_info, [col1, col2, col3]):
        with column:
            port_weights = weights[name]['weights']
            filtered = {k: v for k, v in port_weights.items() if v > 0.001}
            sorted_weights = sorted(filtered.items(), key=lambda x: x[1])
            
            etfs = [item[0] for item in sorted_weights]
            vals = [item[1] for item in sorted_weights]
            
            fig = go.Figure(go.Bar(
                x=vals,
                y=etfs,
                orientation='h',
                marker_color=bar_colors[name],
                text=[f'{v*100:.1f}%' for v in vals],
                textposition='outside',
                textfont=dict(size=10)
            ))
            
            fig.update_layout(
                title=dict(text=f'{name} Portfolio Weights', font=dict(size=13)),
                height=350,
                xaxis_title='Weight',
                yaxis_title='',
                xaxis=dict(tickformat='.0%', range=[0, 0.25]),
                margin=dict(l=50, r=80, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Comparison of Risk Levels Across Metrics")
    
    metrics = ['Return', 'Volatility', 'Max Drawdown', 'Sharpe Ratio']
    risk_levels_names = ['Conservative', 'Balanced', 'Aggressive']
    
    metric_values = {
        'Return': [weights[r]['expected_return'] for r in risk_levels_names],
        'Volatility': [weights[r]['volatility'] for r in risk_levels_names],
        'Max Drawdown': [abs(weights[r]['max_drawdown']) for r in risk_levels_names],
        'Sharpe Ratio': [weights[r]['sharpe_ratio'] for r in risk_levels_names]
    }
    
    fig = go.Figure()
    
    colors = ['#22c55e', '#3b82f6', '#ef4444', '#f59e0b']
    
    x = np.arange(len(risk_levels_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=[pos + (i - 1.5) * width for pos in x],
            y=metric_values[metric],
            width=width,
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title='Comparison of Risk Levels Across Metrics',
        xaxis=dict(
            tickvals=list(x),
            ticktext=risk_levels_names,
            title='Risk Level'
        ),
        yaxis_title='Values',
        barmode='group',
        height=400,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Investment Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Conservative Portfolio")
        st.markdown("""
        **Best for:**
        - Risk-averse investors
        - Near-retirement individuals
        - Capital preservation priority
        
        **Key characteristics:**
        - Heavy allocation to bonds (AGG, BND, HYG)
        - Gold (GLD) for hedging
        - Defensive sectors (XLP, XLV)
        """)
    
    with col2:
        st.markdown("#### Balanced Portfolio")
        st.markdown("""
        **Best for:**
        - Most investors
        - Medium-term investment horizon
        - Balance of growth and stability
        
        **Key characteristics:**
        - Mix of bonds and commodities
        - Moderate equity exposure
        - Best risk-adjusted returns (Sharpe)
        """)
    
    with col3:
        st.markdown("#### Aggressive Portfolio")
        st.markdown("""
        **Best for:**
        - High risk tolerance
        - Long investment horizon
        - Maximum growth seekers
        
        **Key characteristics:**
        - Heavy sector ETFs (XLE, XLI, XLK)
        - Higher volatility accepted
        - Maximum return potential
        """)


def render_planning(data):
    st.markdown("## Investment Planning")
    st.markdown("Turn portfolio theory into actionable investment plans.")
    
    st.markdown("---")
    
    weights = data['weights']
    etf_summary = data['etf_summary']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Your Investment")
        capital = st.number_input(
            "Total Investment Amount (USD)",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0
        )
    
    with col2:
        st.markdown("### Select Risk Level")
        risk_choice = st.radio(
            "Choose your risk strategy:",
            ["Conservative", "Balanced", "Aggressive", "Compare All"],
            horizontal=True
        )
    
    st.markdown("---")
    
    def build_allocation(portfolio_name, capital, weights, etf_summary):
        port_weights = weights[portfolio_name]['weights']
        
        rows = []
        for etf, weight in port_weights.items():
            if weight > 0.001:
                price_row = etf_summary[etf_summary['ETF_Symbol'] == etf]
                if not price_row.empty:
                    price = price_row['Latest_Price'].values[0]
                    name = price_row['Fund_Name'].values[0]
                    amount = capital * weight
                    shares = int(amount / price)
                    used = shares * price
                    rows.append({
                        'ETF': etf,
                        'Name': name,
                        'Weight': weight,
                        'Price': price,
                        'Shares': shares,
                        'Amount': used
                    })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Weight', ascending=False)
        return df
    
    def show_portfolio_plan(portfolio_name, capital, weights, etf_summary, color):
        port = weights[portfolio_name]
        
        expected_value = capital * (1 + port['expected_return'])
        
        st.markdown(f"""
        **Portfolio Summary**
        - Expected Annual Return: **{port['expected_return']*100:.2f}%**
        - Expected Volatility: **{port['volatility']*100:.2f}%**
        - Sharpe Ratio: **{port['sharpe_ratio']:.3f}**
        - Estimated Value After 1 Year: **${expected_value:,.0f}**
        """)
        
        alloc_df = build_allocation(portfolio_name, capital, weights, etf_summary)
        
        total_used = alloc_df['Amount'].sum()
        unused = capital - total_used
        
        st.markdown("### Your Buy List")
        
        display_df = alloc_df.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(1).astype(str) + '%'
        display_df['Price'] = '$' + display_df['Price'].round(2).astype(str)
        display_df['Amount'] = '$' + display_df['Amount'].round(2).astype(str)
        display_df.columns = ['ETF', 'Fund Name', 'Weight', 'Price', 'Shares', 'Amount']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Invested", f"${total_used:,.2f}")
        with col2:
            st.metric("Remaining Cash", f"${unused:,.2f}")
        
        fig = go.Figure(go.Bar(
            x=alloc_df['Amount'],
            y=alloc_df['ETF'],
            orientation='h',
            marker_color=color,
            text=['$' + f'{x:,.0f}' for x in alloc_df['Amount']],
            textposition='outside'
        ))
        fig.update_layout(
            title='Investment Allocation',
            height=max(300, len(alloc_df) * 35),
            xaxis_title='Amount (USD)',
            yaxis_title='',
            margin=dict(l=50, r=100, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if risk_choice == "Compare All":
        tab1, tab2, tab3 = st.tabs(["🟢 Conservative", "🟡 Balanced", "🔴 Aggressive"])
        
        with tab1:
            show_portfolio_plan("Conservative", capital, weights, etf_summary, '#22c55e')
        
        with tab2:
            show_portfolio_plan("Balanced", capital, weights, etf_summary, '#eab308')
        
        with tab3:
            show_portfolio_plan("Aggressive", capital, weights, etf_summary, '#ef4444')
    else:
        colors = {'Conservative': '#22c55e', 'Balanced': '#eab308', 'Aggressive': '#ef4444'}
        st.markdown(f"### {risk_choice} Portfolio Plan")
        show_portfolio_plan(risk_choice, capital, weights, etf_summary, colors[risk_choice])
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: Prices shown are historical reference prices. Actual purchase prices may vary. This is for educational purposes only and does not constitute investment advice.")


def main():
    st.sidebar.markdown("## Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Analysis", "Correlation", "Portfolio", "Planning"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **AlphaOpt** is an intelligent ETF allocation system based on Modern Portfolio Theory.
    
    **Data Period:** 2022-2024  
    **ETFs Analyzed:** 30  
    **Risk Levels:** 3
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built with Streamlit & Plotly*")
    
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all data files are in the 'data' folder.")
        return
    
    if page == "Overview":
        render_overview(data)
    elif page == "Analysis":
        render_analysis(data)
    elif page == "Correlation":
        render_correlation(data)
    elif page == "Portfolio":
        render_portfolio(data)
    elif page == "Planning":
        render_planning(data)


if __name__ == "__main__":
    main()
