import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# Load in data
priceData = pd.read_csv("KRAKEN_ADAUSD, 1D.csv")
print(priceData.head().to_string())
# Algorithms that decide on trades

### RSI-based divergence
def RSIDiv():
    # Detect local maxima (peaks) and minima (valleys) on the 'close' price
    order = 5  # Number of points on each side to compare for local extrema

    priceData['peak'] = False
    priceData['valley'] = False

    # Get indices of local maxima (peaks)
    peak_indices = argrelextrema(priceData['close'].values, np.greater_equal, order=order)[0]
    priceData.loc[peak_indices, 'peak'] = True

    # Get indices of local minima (valleys)
    valley_indices = argrelextrema(priceData['close'].values, np.less_equal, order=order)[0]
    priceData.loc[valley_indices, 'valley'] = True

    # check if there is a RSI-based divergence among the valleys
    valley_points = priceData[priceData['valley']].copy()
    valley_points.reset_index(inplace=True)

    # store the bullish divs
    bullish_divs = []

    # loop over all points
    for i in range(1, len(valley_points)):
        prev = valley_points.iloc[i - 1]
        curr = valley_points.iloc[i]

        price_diff = curr['close'] - prev['close']
        rsi_diff = curr['RSI'] - prev['RSI']

        if price_diff < 0 and rsi_diff > 0:
            bullish_divs.append(curr)

    # transform into dataframe        
    bullish_divs = pd.DataFrame(bullish_divs)

    # check if there is a RSI-based divergence among the peaks
    peak_points = priceData[priceData['peak']].copy()
    peak_points.reset_index(inplace=True)
    
    # Detect bearish divergences on peaks
    bearish_divs = []

    for i in range(1, len(peak_points)):
        prev = peak_points.iloc[i - 1]
        curr = peak_points.iloc[i]

        price_diff = curr['close'] - prev['close']
        rsi_diff = curr['RSI'] - prev['RSI']

        if price_diff > 0 and rsi_diff < 0:
            bearish_divs.append(curr)

    # store the bearish divs
    bearish_divs = pd.DataFrame(bearish_divs)

    # Now pair bullish entries with the *next* bearish exit into a trade
    trades = []

    # loop over all bullish divergences
    for _, entry in bullish_divs.iterrows():

        # Find the first bearish divergence after the entry time
        exits_after_entry = bearish_divs[bearish_divs['time'] > entry['time']]

        # store necessary information
        if not exits_after_entry.empty:
            exit_trade = exits_after_entry.iloc[0]
            trades.append({
                'entry_time': entry['time'],
                'entry_price': entry['close'],
                'exit_time': exit_trade['time'],
                'exit_price': exit_trade['close'],
                'pct_change': (exit_trade['close'] - entry['close']) / entry['close'] * 100
            })

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    return trades_df

def identify_major_highs_lows(df):
    """
    This function identifies major high's and low's, and marks them.
    """

    df = df.copy()
    df['major_low'] = False
    df['major_high'] = False

    closes = df['close'].values

    for i in range(1, len(closes) - 1):  # skip first and last to avoid index errors
        prev_closes = closes[:i]
        curr_close = closes[i]
        next_close = closes[i + 1]

        # Check for major low: current is a new low and next closes higher
        if curr_close < prev_closes.min() and next_close > curr_close:
            df.at[i, 'major_low'] = True

        # Check for major high: current is a new high and next closes lower
        if curr_close > prev_closes.max() and next_close < curr_close:
            df.at[i, 'major_high'] = True

    return df

# add the major highs and lows to the price data
priceData = identify_major_highs_lows(priceData)
major_lows = priceData[priceData['major_low']]
major_highs = priceData[priceData['major_high']]

# call function to get trades based on divergence
RSIDiv_trades = RSIDiv()

# another set of trades created by another function
df =  RSIDiv_trades.copy()

print(df)

## Compare trading strategies
# Add column to classify trade outcome
df['win'] = df['pct_change'] > 0

# Compute summary stats
summary = pd.DataFrame({
    'Total Profit (%)': [df['pct_change'].sum()],
    'Median % Change': [df['pct_change'].median()],
    'Max % Change': [df['pct_change'].max()],
    'Number of Trades': [len(df)],
    'Win Rate (%)': [df['win'].mean() * 100]
})

# Optionally round
summary = summary.round(2)

# Display
print(summary)


## Code to plot the trades on price data

from plotly.subplots import make_subplots

# Create figure with 2 rows: main price chart + RSI subplot
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=["Price with Trades and Bollinger Bands", "RSI and RSI-based MA"]
)

# create a candlestick plot to show price
fig.add_trace(go.Candlestick(
    x=priceData.time,
    open=priceData.open,
    high=priceData.high,
    low=priceData.low,
    close=priceData.close,
    name='Price'
    ), row=1, col=1)

# Add major low markers (triangle-up, green)
fig.add_trace(go.Scatter(
    x=major_lows['time'],
    y=major_lows['close'],
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=10),
    name='Major Low'
))

# Add major high markers (triangle-down, red)
fig.add_trace(go.Scatter(
    x=major_highs['time'],
    y=major_highs['close'],
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=10),
    name='Major High'
))

# --- Bollinger Bands ---
fig.add_trace(go.Scatter(
    x=priceData['time'],
    y=priceData['Upper'],
    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
    name='Upper Band'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=priceData['time'],
    y=priceData['Basis'],
    line=dict(color='rgba(0, 0, 255, 0.4)', width=1),
    name='Basis'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=priceData['time'],
    y=priceData['Lower'],
    line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
    name='Lower Band'
), row=1, col=1)

# --- RSI and RSI MA Subplot ---
fig.add_trace(go.Scatter(
    x=priceData['time'],
    y=priceData['RSI'],
    line=dict(color='orange', width=1),
    name='RSI'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=priceData['time'],
    y=priceData['RSI-based MA'],
    line=dict(color='purple', width=1),
    name='RSI MA'
), row=2, col=1)

# Add overbought/oversold lines
fig.add_shape(type='line', x0=priceData['time'].min(), x1=priceData['time'].max(),
              y0=70, y1=70, line=dict(color='red', dash='dash'), row=2, col=1)
fig.add_shape(type='line', x0=priceData['time'].min(), x1=priceData['time'].max(),
              y0=30, y1=30, line=dict(color='green', dash='dash'), row=2, col=1)

# Add shapes and annotations per trade
for _, trade in trades.iterrows():
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    pct_change = trade['pct_change']

    low_price = min(entry_price, exit_price)
    high_price = max(entry_price, exit_price)

    # 1. Rectangle for trade range
    fig.add_shape(
        type="rect",
        x0=entry_time,
        x1=exit_time,
        y0=low_price,
        y1=high_price,
        fillcolor="rgba(0, 0, 255, 0.1)",
        line=dict(width=0),
        layer="below"
    )

    # 2. Horizontal dotted green line at entry price
    fig.add_shape(
        type="line",
        x0=entry_time,
        x1=exit_time,
        y0=entry_price,
        y1=entry_price,
        line=dict(color="green", dash="dot", width=1.5),
        layer="above"
    )

    # 3. Horizontal dotted red line at exit price
    fig.add_shape(
        type="line",
        x0=entry_time,
        x1=exit_time,
        y0=exit_price,
        y1=exit_price,
        line=dict(color="red", dash="dot", width=1.5),
        layer="above"
    )

    # 4. Vertical dotted lines for timing
    fig.add_shape(
        type="line",
        x0=entry_time,
        x1=entry_time,
        y0=low_price,
        y1=high_price,
        line=dict(color="green", dash="dot", width=1.2),
        layer="above"
    )
    fig.add_shape(
        type="line",
        x0=exit_time,
        x1=exit_time,
        y0=low_price,
        y1=high_price,
        line=dict(color="red", dash="dot", width=1.2),
        layer="above"
    )

    # 5. Invisible marker to show trade info on hover
    mid_time = entry_time  # (optionally: midpoint = entry + (exit - entry)/2)
    mid_price = (entry_price + exit_price) / 2

    hover_text = (
        f"Entry Date: {entry_time}<br>"
        f"Entry Price: {entry_price:.4f}<br>"
        f"Exit Date: {exit_time}<br>"
        f"Exit Price: {exit_price:.4f}<br>"
        f"Change: {pct_change:.2f}%"
    )

    fig.add_trace(go.Scatter(
        x=[mid_time],
        y=[mid_price],
        mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        hovertext=hover_text,
        showlegend=False
    ))

# Final layout
fig.update_layout(
    title='Stock Price with Bollinger Bands and RSI',
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    height=800
)

fig.show()