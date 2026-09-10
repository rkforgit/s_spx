import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="SPX Analysis", layout="wide")

# Sidebar configuration
st.sidebar.header("Data Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
num_top_rows = st.sidebar.slider("Number of Top Rows", min_value=5, max_value=50, value=10, step=5)

# Control for possible low price calculation
target_pct = st.sidebar.number_input(
    "Target Low Change (%)", 
    value=-2.50, 
    step=0.25, 
    format="%.2f"
)

@st.cache_data(ttl=300)
def load_data(start):
    # Download historical SPX data
    spx = yf.download("^GSPC", start=start, auto_adjust=True, multi_level_index=False)
    spx.index = pd.to_datetime(spx.index)

    # Calculate backtest metrics
    spx['LogReturn'] = np.log(spx['Close'] / spx['Close'].shift(1))
    spx['Volatility'] = spx['LogReturn'].rolling(window=5).std() * np.sqrt(252)
    spx['Volatility'] = spx['Volatility'].shift(1)  # Shifted volatility for backtest
    spx['DailyReturn'] = (spx['Close'] - spx['Open']) / spx['Open']

    # Historical ATH and drawdown (using Open prices)
    spx['ATH'] = spx['Open'].cummax()
    spx['Drop_from_ATH_%'] = (spx['Open'] - spx['ATH']) / spx['ATH'] * 100
    
    return spx

@st.cache_data(ttl=60)
def fetch_synthetic_spx():
    try:
        es = yf.Ticker("ES=F")
        es_data = es.history(period="5d", interval="1m")

        spx_ticker = yf.Ticker("^GSPC")
        spx_hist = spx_ticker.history(period="1d")
        spx_close = spx_hist["Close"].iloc[-1]

        es_data = es_data.tz_convert("US/Eastern")
        last_spx_date = spx_hist.index[-1].date()
        ref_time = pd.Timestamp(f"{last_spx_date} 16:00:00", tz="US/Eastern")

        es_ref = es_data.loc[:ref_time].iloc[-1]["Close"]
        synthetic_spx = spx_close * (es_data["Close"] / es_ref)
        pct_change_from_spx = (synthetic_spx - spx_close) / spx_close * 100

        result = pd.DataFrame({
            "Synthetic": synthetic_spx,
            "%": pct_change_from_spx
        })

        latest_synth = result.iloc[-1]
        latest_time = result.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
        return latest_synth["Synthetic"], latest_synth["%"], latest_time
    except Exception as e:
        st.warning(f"Could not calculate Synthetic SPX: {e}")
        return None, None, "N/A"

with st.spinner("Downloading and processing market data..."):
    spx = load_data(start_date)
    synth_price, synth_pct, synth_time = fetch_synthetic_spx()

# -------------------------------------------------------------
# Append Extra Row with Current Day Calculations
# -------------------------------------------------------------
current_volatility = spx['LogReturn'].tail(5).std() * np.sqrt(252)
ath_close = spx['Close'].max()
latest_close = spx['Close'].iloc[-1]
current_drop_ath = (latest_close - ath_close) / ath_close * 100

latest_date = spx.index[-1] + pd.Timedelta(days=1)
extra_row = pd.DataFrame(index=[latest_date], columns=spx.columns)
extra_row.loc[latest_date, 'Volatility'] = current_volatility
extra_row.loc[latest_date, 'Drop_from_ATH_%'] = current_drop_ath

spx_extended = pd.concat([spx, extra_row])

# Calculate possible low price based on sidebar input
possible_low_price = latest_close * (1 + target_pct / 100)

# -------------------------------------------------------------
# Reference Dashboard
# -------------------------------------------------------------
st.subheader("Reference Metrics (Current Day & Synthetic SPX)")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Latest SPX Close", f"${latest_close:,.2f}")
col2.metric("Possible Low Price", f"${possible_low_price:,.2f}", f"{target_pct:+.2f}%")
col3.metric("Today's Volatility", f"{current_volatility:.4f}")
col4.metric("Today's Drop from ATH %", f"{current_drop_ath:.2f}%")

if synth_price is not None:
    col5.metric("Synthetic SPX", f"${synth_price:,.2f}", f"{synth_pct:+.2f}%")
    col6.metric("Synthetic Datetime", synth_time)
else:
    col5.metric("Synthetic SPX", "N/A")
    col6.metric("Synthetic Datetime", "N/A")

st.markdown("---")

# -------------------------------------------------------------
# Filtering & Output Display
# -------------------------------------------------------------
st.subheader(f"Top {num_top_rows} Daily Returns (`Drop_from_ATH_%` > Today's Drop)")

threshold_drop = spx_extended.loc[latest_date, 'Drop_from_ATH_%']

# Filter historical data where Drop_from_ATH_% is HIGHER (closer to 0%) than today's value
filtered_spx = spx[spx['Drop_from_ATH_%'] > threshold_drop].copy()

# Calculate 'Possible Price' based on last SPX close * (1 + DailyReturn)
filtered_spx['Possible Price'] = latest_close * (1 + filtered_spx['DailyReturn'])

# Sort and retrieve top records
top_returns = filtered_spx.nlargest(num_top_rows, 'DailyReturn')

# Display Data Table without Open/Close
display_cols = ['DailyReturn', 'Possible Price', 'Drop_from_ATH_%', 'Volatility']
formatted_df = top_returns[display_cols].copy()

st.dataframe(
    formatted_df.style.format({
        'DailyReturn': "{:.2%}",
        'Possible Price': "${:,.2f}",
        'Drop_from_ATH_%': "{:.2f}%",
        'Volatility': "{:.4f}"
    }),
    use_container_width=True
)

# Visualizing Top Daily Returns
st.subheader("Top Daily Returns Visualization")
st.bar_chart(top_returns['DailyReturn'] * 100)
