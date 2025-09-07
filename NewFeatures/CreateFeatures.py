import numpy as np
import pandas as pd

def detect_candle_patterns(df):
    """
    Detects major candle patterns with clear bullish/bearish/neutral sentiment.

    Input:
        df: DataFrame with columns ['open', 'high', 'low', 'close']

    Output:
        dict of boolean Series for each pattern
    """

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    body = (c - o).abs()
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l
    range_ = h - l
    body_ratio = body / (range_ + 1e-9)  # Avoid div by zero

    # Doji detection thresholds
    doji_thresh = 0.1

    # 1. Engulfing Patterns (need prior candle)
    prev_o = o.shift(1)
    prev_c = c.shift(1)

    bullish_engulfing = (
        (c > o) &  # current candle bullish
        (prev_c < prev_o) &  # previous candle bearish
        (c > prev_o) &
        (o < prev_c)
    )

    bearish_engulfing = (
        (c < o) &  # current candle bearish
        (prev_c > prev_o) &  # previous candle bullish
        (o > prev_c) &
        (c < prev_o)
    )

    # 2. Morning Star and Evening Star (simplified approx)
    # Morning Star: 3 candles - bearish, small body, bullish
    ms_cond1 = (prev_c < prev_o)  # 2nd last bearish
    ms_cond2 = (body.shift(1) < body.shift(2) * 0.5)  # middle candle small body
    ms_cond3 = (c > o)  # last candle bullish
    morning_star = ms_cond1 & ms_cond2 & ms_cond3

    # Evening Star: opposite of morning star
    es_cond1 = (prev_c > prev_o)  # 2nd last bullish
    es_cond2 = (body.shift(1) < body.shift(2) * 0.5)
    es_cond3 = (c < o)
    evening_star = es_cond1 & es_cond2 & es_cond3

    # 3. Hammer and Hanging Man
    hammer = (
        (body_ratio < 0.3) &
        (lower_shadow >= 2 * body) &
        (upper_shadow <= body)
    )
    hanging_man = hammer & (prev_c > prev_o)  # bearish signal

    # 4. Inverted Hammer and Shooting Star
    inverted_hammer = (
        (body_ratio < 0.3) &
        (upper_shadow >= 2 * body) &
        (lower_shadow <= body)
    )
    shooting_star = inverted_hammer & (prev_c < prev_o)  # bearish signal

    # 5. Doji types
    doji = body_ratio < doji_thresh

    # Gravestone Doji: doji + long upper shadow, little/no lower shadow
    gravestone_doji = doji & (upper_shadow >= 2 * body) & (lower_shadow <= body * 0.1)

    # Dragonfly Doji: doji + long lower shadow, little/no upper shadow
    dragonfly_doji = doji & (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.1)

    # Standard Doji: doji but neither gravestone nor dragonfly
    standard_doji = doji & ~(gravestone_doji | dragonfly_doji)

    # simple green (higher close than open) or red indicators
    green = c >= o
    red = o >= c

    # return a dictionary with all the new features
    return pd.DataFrame({
        'green' : green.fillna(False),
        'red' : red.fillna(False),
        'bullish_engulfing': bullish_engulfing.fillna(False),
        'bearish_engulfing': bearish_engulfing.fillna(False),
        'morning_star': morning_star.fillna(False),
        'evening_star': evening_star.fillna(False),
        'hammer': hammer.fillna(False),
        'hanging_man': hanging_man.fillna(False),
        'inverted_hammer': inverted_hammer.fillna(False),
        'shooting_star': shooting_star.fillna(False),
        'dragonfly_doji': dragonfly_doji.fillna(False),
        'gravestone_doji': gravestone_doji.fillna(False),
        'standard_doji': standard_doji.fillna(False),
    })

def get_price_pivots(df, span=2):
    """
    Returns boolean lists indicating pivot highs (peaks) and pivot lows (valleys)
    for each row in the DataFrame.

    A pivot high is a local maximum where the high is greater than the highs 
    of `span` bars before and after.
    
    A pivot low is a local minimum where the low is less than the lows 
    of `span` bars before and after.

    Args:
        df (pd.DataFrame): Must contain 'high' and 'low' columns.
        span (int): Number of bars before and after to consider.

    Returns:
        (pivot_high, pivot_low): Tuple of boolean lists of length len(df)
    """
    n = len(df)
    pivot_high = [False] * n
    pivot_low = [False] * n

    for i in range(span, n - span):
        high_window = df['high'].iloc[i - span: i + span + 1]
        low_window = df['low'].iloc[i - span: i + span + 1]

        center_high = df['high'].iloc[i]
        center_low = df['low'].iloc[i]

        if center_high == max(high_window) and list(high_window).count(center_high) == 1:
            pivot_high[i] = True

        if center_low == min(low_window) and list(low_window).count(center_low) == 1:
            pivot_low[i] = True

    return pivot_high, pivot_low