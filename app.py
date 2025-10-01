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

# --- Calculation Functions ---

def calculate_binomial_prices(S, K, r_percent, vol_percent, T):
    """Calculates single-period binomial option prices."""
    results = {}
    r, sigma = r_percent / 100, vol_percent / 100
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

def black_scholes(S, K, T, r_percent, vol_percent):
    """Calculates Black-Scholes option prices and Greeks."""
    results = {}
    r, sigma = r_percent / 100, vol_percent / 100
    
    if T <= 0 or sigma <= 0:
        results['error'] = "Time to expiration and volatility must be positive for Black-Scholes."
        return results

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    call_theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    results.update({
        'call_price': call_price, 'put_price': put_price,
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
    call_payoff = np.maximum(spot_range - K, 0)
    put_payoff = np.maximum(K - spot_range, 0)
    
    call_profit = call_payoff - call_price
    put_profit = put_payoff - put_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_profit, mode='lines', name='Call Profit/Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=spot_range, y=put_profit, mode='lines', name='Put Profit/Loss', line=dict(color='red')))
    
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=K, line_dash="dash", line_color="grey", annotation_text=f"Strike Price: {K}")
    
    fig.update_layout(
        title="Option Profit/Loss at Expiration",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss",
        legend_title="Option Type"
    )
    return fig

# --- Streamlit UI ---
st.title("📈 Options Analysis Dashboard for Nifty 50")

# --- Sidebar ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.radio("Select Pricing Model", ["Black-Scholes", "Binomial Model (Single Step)"], horizontal=True)
ticker = st.sidebar.selectbox("Select Nifty 50 Stock", NIFTY_50_TICKERS)

# --- Data Fetching ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    live_price = info.get('regularMarketPrice', info.get('previousClose'))
    hist = stock.history(period="1y")
    
    if live_price is None:
        st.sidebar.error("Could not fetch live price.")
        st.stop()

    hist_vol = calculate_historical_volatility(hist)
    expirations = stock.options
    
    # FIX: Check if expirations tuple is empty. If so, stop the app.
    if not expirations:
        st.sidebar.warning(f"No options data available for {ticker}.")
        st.warning(f"Could not retrieve any option expiration dates for {ticker}. Please select another stock.")
        st.stop()

    st.sidebar.metric(label=f"Current Price ({ticker})", value=f"₹{live_price:,.2f}")
    st.sidebar.metric(label="1-Year Historical Volatility", value=f"{hist_vol:.2f}%")

except Exception as e:
    st.sidebar.error(f"Failed to fetch data: {e}")
    st.stop()

# --- User Inputs in Sidebar ---
exp_date = st.sidebar.selectbox("Expiration Date", expirations)
time_to_exp_days = (pd.to_datetime(exp_date).date() - date.today()).days
time_to_exp_years = time_to_exp_days / 365.0

default_strike = float(round(live_price / 10) * 10)
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=default_strike, step=5.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf) %", min_value=0.0, value=7.0, step=0.1)
volatility = st.sidebar.number_input("Implied Volatility (σ) %", min_value=0.1, value=round(hist_vol, 2), step=0.5)

# --- Main Page Layout ---
col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.subheader(f"Historical Price for {ticker}")
    st.line_chart(hist['Close'])

with col_data:
    st.subheader("Model Calculation")
    results = {}
    if model_choice == "Black-Scholes":
        results = black_scholes(live_price, strike_price, time_to_exp_years, risk_free_rate, volatility)
    else:
        results = calculate_binomial_prices(live_price, strike_price, risk_free_rate, volatility, time_to_exp_years)

    if 'error' in results and results['error']:
        st.error(results['error'])
    else:
        if 'warning' in results: st.warning(results['warning'])
        
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

# --- Payoff Diagram ---
if 'error' not in results or not results['error']:
    st.subheader("Profit/Loss Analysis")
    st.plotly_chart(plot_payoff_diagram(live_price, strike_price, results.get('call_price', 0), results.get('put_price', 0)), use_container_width=True)

# --- Option Chain Data ---
st.subheader(f"Live Option Chain for {exp_date}")
try:
    opt_chain = stock.option_chain(exp_date)
    calls, puts = opt_chain.calls, opt_chain.puts
    
    calls_styled = calls.style.applymap(lambda x: 'background-color: #e8f3e8', subset=pd.IndexSlice[calls['inTheMoney'] == True, :])
    puts_styled = puts.style.applymap(lambda x: 'background-color: #e8f3e8', subset=pd.IndexSlice[puts['inTheMoney'] == True, :])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### CALLS")
        st.dataframe(calls_styled)
    with c2:
        st.markdown("##### PUTS")
        st.dataframe(puts_styled)
except Exception:
    st.warning("Could not retrieve the option chain for this expiration date.")

st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: This is for educational purposes only. Not financial advice.</p>", unsafe_allow_html=True)

