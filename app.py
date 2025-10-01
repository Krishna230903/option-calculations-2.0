import streamlit as st
import yfinance as yf
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.stats import norm
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Options Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- NIFTY 50 Tickers ---
NIFTY_50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'LICI.NS',
    'BAJFINANCE.NS', 'HCLTECH.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
    'MARUTI.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 'TITAN.NS',
    'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
    'M&M.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'POWERGRID.NS',
    'BAJAJFINSV.NS', 'COALINDIA.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'HINDALCO.NS',
    'TECHM.NS', 'GRASIM.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'DRREDDY.NS',
    'SBILIFE.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'BRITANNIA.NS', 'APOLLOHOSP.NS',
    'SHRIRAMFIN.NS', 'HDFCLIFE.NS', 'BAJAJ-AUTO.NS', 'BPCL.NS', 'LTIM.NS'
]

# --- Cached Data Fetching Functions ---
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(ticker_str):
    stock = yf.Ticker(ticker_str)
    info = stock.info
    hist = stock.history(period="1y")
    expirations = stock.options
    return info, hist, expirations

@st.cache_data(ttl=600)
def get_option_chain(ticker_str, exp_date):
    return yf.Ticker(ticker_str).option_chain(exp_date)

# --- Calculation Functions ---

def calculate_binomial_prices(S, K, r_percent, vol_percent, T):
    """Calculates single-period binomial option prices."""
    results = {}
    r, sigma = r_percent / 100, vol_percent / 100
    if T <= 0: T = 0.0001 # Avoid division by zero
    u, d = 1 + sigma, 1 / (1 + sigma)
    R = (1 + r)**T
    if not (d < R < u):
        results['warning'] = f"Arbitrage opportunity may exist. R ({R:.4f}) is not between d ({d:.4f}) and u ({u:.4f})."
    p = (R - d) / (u - d)
    su, sd = S * u, S * d
    cu, cd = max(0, su - K), max(0, sd - K)
    pu, pd = max(0, K - su), max(0, K - sd)
    call_price = (p * cu + (1 - p) * cd) / R
    put_price = (p * pu + (1 - p) * pd) / R
    results.update({'call_price': call_price, 'put_price': put_price})
    return results

def black_scholes(S, K, T, r_percent, vol_percent, option_type='call'):
    """Calculates Black-Scholes option prices and Greeks."""
    results = {}
    r, sigma = r_percent / 100, vol_percent / 100

    if T <= 0 or sigma <= 0:
        if option_type == 'call': return {'call_price': max(0, S - K)}
        if option_type == 'put': return {'put_price': max(0, K - S)}
        return {}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        results['call_price'] = price
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        results['put_price'] = price
    else:
        return {}
    
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    call_theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    results.update({
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'vega': vega,
        'call_theta': call_theta, 'put_theta': put_theta
    })
    return results

def calculate_historical_volatility(data):
    """Calculates annualized historical volatility."""
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    daily_vol = log_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol * 100

def plot_payoff_diagram(S, K, call_price, put_price):
    """Creates an interactive payoff diagram using Plotly."""
    spot_range = np.linspace(S * 0.75, S * 1.25, 100)
    call_profit = np.maximum(spot_range - K, 0) - call_price
    put_profit = np.maximum(K - spot_range, 0) - put_price

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_profit, mode='lines', name='Call Profit/Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=spot_range, y=put_profit, mode='lines', name='Put Profit/Loss', line=dict(color='red')))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=K, line_dash="dash", line_color="grey", annotation_text=f"Strike Price: {K}")
    fig.update_layout(title="Option Profit/Loss at Expiration", xaxis_title="Stock Price", yaxis_title="Profit / Loss")
    return fig

# --- Streamlit UI ---
st.title("📈 Options Analysis Dashboard")

# --- Sidebar ---
st.sidebar.header("Configuration")
custom_ticker = st.sidebar.text_input("Enter Custom Ticker (e.g., 'AAPL')").upper()
nifty_ticker = st.sidebar.selectbox("Or Select Nifty 50 Stock", NIFTY_50_TICKERS)
ticker = custom_ticker if custom_ticker else nifty_ticker

model_choice = st.sidebar.radio("Select Pricing Model", ["Black-Scholes", "Binomial Model (Single Step)"], horizontal=True)

try:
    info, hist, expirations = get_stock_data(ticker)
    live_price = info.get('regularMarketPrice', info.get('previousClose'))
    if live_price is None:
        st.sidebar.error(f"Could not fetch live price for {ticker}.")
        st.stop()
    hist_vol = calculate_historical_volatility(hist)

    if not expirations:
        st.sidebar.warning(f"No options data available for {ticker}.")
        st.warning("This can happen outside market hours or for certain stocks.")
        st.stop()

    st.sidebar.metric(label=f"Current Price ({ticker})", value=f"₹{live_price:,.2f}" if info.get('currency') == 'INR' else f"${live_price:,.2f}")
    st.sidebar.metric(label="1-Year Historical Volatility", value=f"{hist_vol:.2f}%")

except Exception as e:
    st.sidebar.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
    st.stop()

# --- User Inputs in Sidebar ---
exp_date = st.sidebar.selectbox("Expiration Date", expirations)
time_to_exp_days = (pd.to_datetime(exp_date).date() - date.today()).days
time_to_exp_years = time_to_exp_days / 365.0

default_strike = float(round(live_price / 10) * 10)
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=default_strike, step=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf) %", min_value=0.0, value=7.0, step=0.1)
volatility = st.sidebar.number_input("Volatility (σ) %", min_value=0.1, value=round(hist_vol, 2), step=0.5)

# --- Main Page Layout with Tabs ---
tab1, tab2 = st.tabs(["Analysis & Payoff", "Live Option Chain"])

with tab1:
    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        st.subheader(f"Historical Price for {ticker}")
        st.line_chart(hist['Close'])

    with col_data:
        st.subheader("Model Calculation")
        results = {}
        if model_choice == "Black-Scholes":
            call_res = black_scholes(live_price, strike_price, time_to_exp_years, risk_free_rate, volatility, 'call')
            put_res = black_scholes(live_price, strike_price, time_to_exp_years, risk_free_rate, volatility, 'put')
            results = {**call_res, **put_res}
        else:
            results = calculate_binomial_prices(live_price, strike_price, risk_free_rate, volatility, time_to_exp_years)

        c1, c2 = st.columns(2)
        c1.metric("Call Option Value", f"₹{results.get('call_price', 0):.2f}")
        c2.metric("Put Option Value", f"₹{results.get('put_price', 0):.2f}")

        if model_choice == "Black-Scholes":
            st.markdown("---")
            st.markdown("##### Option Greeks")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Call Delta", f"{results.get('call_delta', 0):.4f}")
            g2.metric("Put Delta", f"{results.get('put_delta', 0):.4f}")
            g1.metric("Gamma", f"{results.get('gamma', 0):.4f}")
            g2.metric("Vega", f"{results.get('vega', 0):.4f}")
            g3.metric("Call Theta", f"{results.get('call_theta', 0):.4f}")
            g4.metric("Put Theta", f"{results.get('put_theta', 0):.4f}")

    st.subheader("Profit/Loss Analysis")
    st.plotly_chart(plot_payoff_diagram(live_price, strike_price, results.get('call_price', 0), results.get('put_price', 0)), use_container_width=True)

with tab2:
    st.subheader(f"Live Option Chain for {exp_date}")
    try:
        opt_chain = get_option_chain(ticker, exp_date)
        calls, puts = opt_chain.calls.copy(), opt_chain.puts.copy()
        
        # Calculate Model Price for the entire chain
        calls['Model Price'] = calls.apply(
            lambda row: black_scholes(live_price, row['strike'], time_to_exp_years, risk_free_rate, row['impliedVolatility']*100, 'call').get('call_price', 0), axis=1
        )
        puts['Model Price'] = puts.apply(
            lambda row: black_scholes(live_price, row['strike'], time_to_exp_years, risk_free_rate, row['impliedVolatility']*100, 'put').get('put_price', 0), axis=1
        )
        
        # Find ATM strike
        atm_strike = calls.iloc[(calls['strike'] - live_price).abs().argsort()[:1]].iloc[0]['strike']

        def style_chain(df, atm_strike):
            style = df.style.applymap(lambda x: 'background-color: #e8f3e8', subset=pd.IndexSlice[df['inTheMoney'] == True, :])\
                           .apply(lambda x: ['background: #e0f2fe' if x.strike == atm_strike else '' for i in x], axis=1)
            return style

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### CALLS")
            st.dataframe(style_chain(calls, atm_strike))
        with c2:
            st.markdown("##### PUTS")
            st.dataframe(style_chain(puts, atm_strike))
            
    except Exception as e:
        st.warning(f"Could not retrieve the option chain for this date. Error: {e}")

st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: This is for educational purposes only. Not financial advice.</p>", unsafe_allow_html=True)

