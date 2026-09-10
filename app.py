import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="SPX Backtest & Analysis", layout="wide")

st.title("📈 SPX Backtest & Current Market Analysis")

# Sidebar configuration
st.sidebar.header("Data Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
num_top_rows = st.sidebar.slider("Number of Top Rows", min_value=5, max_value=50, value=10, step=5)

@st.cache_data(ttl=3600)
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

with st.spinner("Downloading and processing market data..."):
    spx = load_data(start_date)

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

# -------------------------------------------------------------
# Reference Dashboard
# -------------------------------------------------------------
st.subheader("Reference Metrics (Current Day / Extra Row)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close Price", f"${latest_close:,.2f}")
col2.metric("Today's Volatility (Unshifted)", f"{current_volatility:.4f}")
col3.metric("Today's Drop from ATH % (Close-based)", f"{current_drop_ath:.2f}%")
col4.metric("Extra Row Date", latest_date.strftime("%Y-%m-%d"))

st.markdown("---")

# -------------------------------------------------------------
# Filtering & Output Display
# -------------------------------------------------------------
st.subheader(f"Top {num_top_rows} Daily Returns (`Drop_from_ATH_%` > Today's Drop)")

threshold_drop = spx_extended.loc[latest_date, 'Drop_from_ATH_%']

# Filter historical data where Drop_from_ATH_% is HIGHER (closer to 0%) than today's value
filtered_spx = spx[spx['Drop_from_ATH_%'] > threshold_drop]

# Sort and retrieve top records
top_returns = filtered_spx.nlargest(num_top_rows, 'DailyReturn')

# Display Data Table with Formatted Percentages
display_cols = ['Open', 'Close', 'DailyReturn', 'Drop_from_ATH_%', 'Volatility']
formatted_df = top_returns[display_cols].copy()

st.dataframe(
    formatted_df.style.format({
        'Open': "${:,.2f}",
        'Close': "${:,.2f}",
        'DailyReturn': "{:.2%}",
        'Drop_from_ATH_%': "{:.2f}%",
        'Volatility': "{:.4f}"
    }),
    use_container_width=True
)

# Visualizing Top Daily Returns
st.subheader("Top Daily Returns Visualization")
st.bar_chart(top_returns['DailyReturn'] * 100)