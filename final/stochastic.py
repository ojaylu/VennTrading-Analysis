

def get_stoch_osc(high, low, close, k_lookback, d_lookback):
        lowest_low = low.rolling(k_lookback).min()
        highest_high = high.rolling(k_lookback).max()
        k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_line = k_line.rolling(d_lookback).mean()
        return k_line, d_line