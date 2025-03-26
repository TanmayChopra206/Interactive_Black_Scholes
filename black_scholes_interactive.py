import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq  # More robust root finder than newton
import plotly.graph_objects as go
import math


# ===========================================
# Black-Scholes Formulas & Greeks
# ===========================================

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price.

    Args:
        S (float): Current stock price
        K (float): Option strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized decimal)
        sigma (float): Volatility (annualized decimal)
        option_type (str): 'call' or 'put'

    Returns:
        float: Option price, or NaN if inputs are invalid (e.g., T=0, sigma=0)
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases: Option price is intrinsic value at expiration
        if option_type == 'call':
            return max(0, S - K)
        elif option_type == 'put':
            return max(0, K - S)
        else:
            return np.nan  # Should not happen with valid inputs

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            st.error("Option type must be 'call' or 'put'")
            return np.nan
        return price
    except Exception as e:
        st.error(f"Error in Black-Scholes calculation: {e}")
        return np.nan


def delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    else:
        return np.nan


def gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    # Returns Vega per 1% change in vol
    if T <= 0 or sigma <= 0 or S <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # Vega is typically quoted as change per 1% point, hence divide by 100
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S, K, T, r, sigma, option_type='call'):
    # Returns Theta per day
    if T <= 0 or sigma <= 0 or S <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_yr = term1 + term2
    elif option_type == 'put':
        term2 = + r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_yr = term1 + term2
    else:
        return np.nan
    # Theta is typically quoted per calendar day, hence divide by 365
    return theta_yr / 365.0


def rho(S, K, T, r, sigma, option_type='call'):
    # Returns Rho per 1% change in rate
    if T <= 0 or sigma <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        # Rho is typically quoted per 1% point change in r, hence divide by 100
        rho_val = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    elif option_type == 'put':
        rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
    else:
        return np.nan
    return rho_val


# ===========================================
# Implied Volatility Calculation
# ===========================================

def implied_volatility(market_price, S, K, T, r, option_type='call', low_vol=1e-5, high_vol=5.0):
    """
    Calculates the implied volatility using the Brentq algorithm.

    Args:
        market_price (float): Observed market price of the option
        S, K, T, r: Black-Scholes parameters
        option_type (str): 'call' or 'put'
        low_vol (float): Lower bound for volatility search
        high_vol (float): Upper bound for volatility search

    Returns:
        float: Implied volatility (decimal), or NaN if not found or price is invalid.
    """
    if T <= 0: return np.nan  # Cannot calculate IV for expired options
    if option_type == 'call' and market_price < max(0, S - K * np.exp(-r * T)):
        st.warning("Market price is below intrinsic value for call. IV cannot be calculated.", icon="⚠️")
        return np.nan
    if option_type == 'put' and market_price < max(0, K * np.exp(-r * T) - S):
        st.warning("Market price is below intrinsic value for put. IV cannot be calculated.", icon="⚠️")
        return np.nan

    # Define the objective function: difference between BS price and market price
    def objective_func(sigma):
        price = black_scholes(S, K, T, r, sigma, option_type)
        # Handle cases where black_scholes returns nan due to extreme sigma
        if np.isnan(price):
            # If sigma is very low, price -> intrinsic; if very high, price -> S (call) or K*exp(-rT) (put)
            # Return a large difference to push the solver away
            return 1e6  # Return a large number if calculation fails
        return price - market_price

    try:
        # Check if the objective function changes sign over the interval
        f_low = objective_func(low_vol)
        f_high = objective_func(high_vol)

        if np.sign(f_low) == np.sign(f_high):
            # Try expanding the range or checking edge cases
            if abs(f_low) < 1e-4: return low_vol  # Price very close at low vol
            if abs(f_high) < 1e-4: return high_vol  # Price very close at high vol
            st.warning(
                f"IV search failed: Black-Scholes price does not bracket market price in range [{low_vol * 100:.1f}%, {high_vol * 100:.1f}%]. BS({low_vol:.3f})={objective_func(low_vol) + market_price:.3f}, BS({high_vol:.3f})={objective_func(high_vol) + market_price:.3f}. Market={market_price:.3f}",
                icon="⚠️")
            return np.nan

        # Use Brentq to find the root (sigma)
        iv = brentq(objective_func, low_vol, high_vol, xtol=1e-6, rtol=1e-6)
        return iv

    except ValueError as ve:
        # This often happens if f(a) and f(b) have the same sign
        st.warning(
            f"Implied Volatility calculation failed (ValueError): {ve}. Check if market price [{market_price:.3f}] is achievable within vol range [{low_vol * 100:.1f}%-{high_vol * 100:.1f}%].",
            icon="⚠️")
        return np.nan
    except Exception as e:
        st.error(f"An unexpected error occurred during IV calculation: {e}")
        return np.nan


# ===========================================
# Plotting Function
# ===========================================

def plot_greeks(greek_func, S, K, T_days, r_pct, sigma_pct, option_type, plot_vs='Stock Price'):
    """ Generates Plotly figure for a Greek vs. Stock Price or Time """
    T = T_days / 365.0
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0
    greek_name = greek_func.__name__.capitalize()

    fig = go.Figure()

    if plot_vs == 'Stock Price':
        S_range = np.linspace(max(0.1, S * 0.7), S * 1.3, 100)  # Range around current S
        T_plot = T
        greek_values = [greek_func(s_val, K, T_plot, r, sigma, option_type) if greek_name != 'Gamma'
                        else greek_func(s_val, K, T_plot, r, sigma)
                        for s_val in S_range]
        x_values = S_range
        x_title = "Stock Price ($)"
        current_val_marker = S

    elif plot_vs == 'Time (Days)':
        T_days_range = np.linspace(max(1, T_days * 0.05), T_days, 100)  # Range from near 0 to current T
        S_plot = S
        greek_values = [greek_func(S_plot, K, t_val / 365.0, r, sigma, option_type) if greek_name != 'Gamma'
                        else greek_func(S_plot, K, t_val / 365.0, r, sigma)
                        for t_val in T_days_range]
        x_values = T_days_range
        x_title = "Time to Expiration (Days)"
        current_val_marker = T_days

    else:  # Should not happen
        return fig  # Return empty figure

    # Remove NaNs which can occur at T=0 or S=0
    valid_indices = ~np.isnan(greek_values)
    x_values = np.array(x_values)[valid_indices]
    greek_values = np.array(greek_values)[valid_indices]

    if len(x_values) > 0:
        fig.add_trace(go.Scatter(x=x_values, y=greek_values, mode='lines', name=greek_name))

        # Add marker for current value
        current_greek = greek_func(S, K, T, r, sigma, option_type) if greek_name != 'Gamma' else greek_func(S, K, T, r,
                                                                                                            sigma)
        if not np.isnan(current_greek):
            fig.add_trace(go.Scatter(x=[current_val_marker], y=[current_greek], mode='markers',
                                     marker=dict(color='red', size=10), name='Current'))

    fig.update_layout(
        title=f'{greek_name} vs. {plot_vs}',
        xaxis_title=x_title,
        yaxis_title=greek_name,
        legend_title="Trace"
    )
    return fig


# ===========================================
# Streamlit App Layout
# ===========================================

st.set_page_config(layout="wide")
st.title("📈 Black-Scholes Option Pricing & Greeks Calculator")
st.markdown("Calculate option prices, Greeks, and implied volatility using the Black-Scholes model.")

# --- Input Columns ---
col1, col2 = st.columns([1, 1])  # Input columns

with col1:
    st.subheader("Option Parameters")
    option_type = st.radio("Option Type", ('Call', 'Put'), horizontal=True)
    K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    T_days = st.slider("Time to Expiration (Days)", min_value=0, max_value=1095, value=30, step=1)  # Max 3 years
    # Convert days to years for calculations, handle T=0 edge case for BSM
    T = max(T_days, 1e-6) / 365.0  # Avoid T=0 directly for formulas, use small epsilon

with col2:
    st.subheader("Market Parameters")
    S = st.number_input("Underlying Price (S)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    r_pct = st.slider("Risk-Free Rate (r %)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    # For BS calculation, allow specifying vol OR calculating IV
    vol_calc_method = st.radio("Volatility Input", ('Specify Volatility', 'Calculate Implied Volatility'),
                               horizontal=True, index=0)

    sigma_pct = None
    iv_pct = None  # Initialize Implied Volatility result

    if vol_calc_method == 'Specify Volatility':
        sigma_pct = st.slider("Volatility (σ %)", min_value=0.1, max_value=200.0, value=20.0, step=0.5)
        sigma = sigma_pct / 100.0
        market_price_input = None  # Not needed for specifying vol
    else:  # Calculate Implied Volatility
        market_price_input = st.number_input("Market Price of Option", min_value=0.0, value=1.50, step=0.01,
                                             format="%.2f")
        if S > 0 and K > 0 and T > 0 and market_price_input is not None:
            iv = implied_volatility(market_price_input, S, K, T, r_pct / 100.0, option_type.lower())
            if iv is not None and not np.isnan(iv):
                iv_pct = iv * 100.0
                sigma = iv  # Use calculated IV for Greeks
                st.success(f"Implied Volatility (σ %): {iv_pct:.2f}%")
            else:
                # IV calc failed, cannot calculate price/greeks reliably based on market price
                sigma = np.nan  # Indicate sigma is unknown/invalid
                sigma_pct = np.nan
                st.warning("Cannot calculate price/Greeks without valid volatility.", icon="⚠️")
        else:
            sigma = np.nan  # Not enough info for IV calc
            sigma_pct = np.nan

# --- Calculations (only if sigma is valid) ---
price, delta_val, gamma_val, vega_val, theta_val, rho_val = (np.nan,) * 6  # Initialize results as NaN

if sigma is not None and not np.isnan(sigma):
    r = r_pct / 100.0  # Ensure rate is decimal

    # Calculate Price
    price = black_scholes(S, K, T, r, sigma, option_type.lower())

    # Calculate Greeks (handle potential NaNs from formulas)
    delta_val = delta(S, K, T, r, sigma, option_type.lower())
    gamma_val = gamma(S, K, T, r, sigma)
    vega_val = vega(S, K, T, r, sigma)
    theta_val = theta(S, K, T, r, sigma, option_type.lower())
    rho_val = rho(S, K, T, r, sigma, option_type.lower())

    # If IV was calculated, price should match market_price_input (theoretically)
    # If vol was specified, price is the theoretical price

# --- Output Columns ---
st.markdown("---")
st.subheader("Results")

# Handle display based on calculation method
if vol_calc_method == 'Specify Volatility':
    st.metric(label=f"Theoretical {option_type} Price", value=f"${price:.3f}" if not np.isnan(price) else "N/A")
    st.caption(
        f"Calculated using specified Volatility = {sigma_pct:.2f}%" if sigma_pct is not None else "Volatility not specified")
else:  # Implied Volatility was calculated
    st.metric(label=f"Market Price ({option_type})",
              value=f"${market_price_input:.3f}" if market_price_input is not None else "N/A")
    if iv_pct is not None:
        st.caption(f"Implied Volatility = {iv_pct:.2f}%")
    else:
        st.caption("Implied Volatility could not be calculated.")

st.markdown("---")
st.subheader("Option Greeks")
gc1, gc2, gc3, gc4, gc5 = st.columns(5)
gc1.metric(label="Delta", value=f"{delta_val:.4f}" if not np.isnan(delta_val) else "N/A")
gc2.metric(label="Gamma", value=f"{gamma_val:.4f}" if not np.isnan(gamma_val) else "N/A")
gc3.metric(label="Vega", value=f"{vega_val:.4f}" if not np.isnan(vega_val) else "N/A",
           help="Price change per 1% change in Volatility")
gc4.metric(label="Theta", value=f"{theta_val:.4f}" if not np.isnan(theta_val) else "N/A",
           help="Price change per 1 day decrease in time")
gc5.metric(label="Rho", value=f"{rho_val:.4f}" if not np.isnan(rho_val) else "N/A",
           help="Price change per 1% change in Risk-Free Rate")

# --- Visualization Section ---
st.markdown("---")
st.subheader("Greeks Visualization")

if sigma is not None and not np.isnan(sigma):  # Only plot if sigma is valid
    plot_col1, plot_col2 = st.columns([1, 3])

    with plot_col1:
        greek_to_plot_name = st.selectbox("Select Greek", ("Delta", "Gamma", "Vega", "Theta", "Rho"))
        plot_variable = st.radio("Plot Against", ('Stock Price', 'Time (Days)'), index=0)

    with plot_col2:
        greek_functions = {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }
        selected_greek_func = greek_functions.get(greek_to_plot_name)

        if selected_greek_func:
            fig = plot_greeks(selected_greek_func, S, K, T_days, r_pct, sigma * 100.0, option_type.lower(),
                              plot_variable)  # Pass sigma_pct back
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selected Greek function not found.")
else:
    st.warning("Cannot visualize Greeks without a valid volatility value.", icon="⚠️")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app uses the Black-Scholes model to calculate option prices and Greeks. "
    "It can also compute Implied Volatility from a market price. "
    "Use for educational purposes only. Financial models have limitations."
)
