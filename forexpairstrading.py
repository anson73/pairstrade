import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt

def get_historical_Data(tickers, start, end):
    data = pd.DataFrame()
    returns = pd.DataFrame()
    for instrument in tickers:
        prices = yf.download(instrument, start, end)
        data[instrument] = prices['Close']
        returns[instrument] = np.append(data[instrument][1:].reset_index(drop=True)/data[instrument][:-1].reset_index(drop=True) - 1 , 0)
    
    return data

def get_pairs_info():
    major_pairs = ['EUR=X','GBP=X','AUD=X','NZD=X','JPY=X','CHF=X','CAD=X']
    start = '2014-01-01'
    end = '2024-01-01'
    data = get_historical_Data(major_pairs, start, end)

    # Get correlation
    print(data.corr())

    # Get Co-integration/Stationary P-Values
    for x in major_pairs:
        for y in major_pairs:
            print(f"({x}, {y})", end=" ")
            if x == y:
                print("0")
            else:
                print(adfuller(data[x] - data[y])[1])
        print("")

def audnzd_strategy():
    major_pairs = ['AUD=X','NZD=X']
    start = '2014-01-01'
    end = '2024-01-01'
    data = get_historical_Data(major_pairs, start, end)

    a = "AUD=X"
    b = "NZD=X"

    x = 0
    y = 1512 # 252 trading days per year * 6 years = 1512
    hold_z_score = None
    while y < 2604:
        spread_lookback = data.iloc[x:y][b] - data.iloc[x:y][a]
        z_scores = (spread_lookback - spread_lookback.mean())/spread_lookback.std()

        spread_current = data.iloc[y+1][b] - data.iloc[y+1][a]
        z_score_current = (spread_current - spread_lookback.mean())/spread_lookback.std()
        
        if (hold_z_score is not None) and (hold_z_score * z_score_current < 0):
            print(f"EXIT Trade at: AUD={data.iloc[y+1][a].round(4)}, NZD={data.iloc[y+1][b].round(4)}, "
                  f"Date={str(data.iloc[y+1,[1]])[23:33]} because z-score is: {z_score_current.round(4)}\n")
            
            hold_z_score = None

        elif (hold_z_score is None) and (z_score_current >= 2 or z_score_current <= -2):
            print(f"ENTER Trade at: AUD={data.iloc[y+1][a].round(4)}, NZD={data.iloc[y+1][b].round(4)}, "
                  f"Date={str(data.iloc[y+1,[1]])[23:33]} because z-score is: {z_score_current.round(4)}")
            
            if z_score_current <= -2:
                print("(LONG AUD, SHORT NZD)")
            elif z_score_current >= 2:
                print("(LONG NZD, SHORT AUD)")
            hold_z_score = z_score_current
            
        x += 1
        y += 1

def audnzd_info():
    major_pairs = ['AUD=X','NZD=X']
    start = '2014-01-01'
    end = '2024-01-01'
    data = get_historical_Data(major_pairs, start, end)
    
    a = "AUD=X"
    b = "NZD=X"

    # Co-integration test using Engle-Granger method
    p_value = coint(data[a], data[b])[1]
    print('Engle-Granger co-integration test: ', p_value)

    # Co-integration test using Augmented Dickey-Fuller Test
    print('ADF co-integration test (Spread): ', adfuller(data[b] - data[a])[1])
    # print('ADF co-integration test (Ratio): ', adfuller(data[b] / data[a])[1])

    spread = data[b] - data[a]
    z_score = (spread - spread.mean())/spread.std()

    # Plot spread between NZDUSD and AUDUSD
    plt.figure(figsize=(8, 6), dpi=200)
    spread = data[b] - data[a]
    plt.plot(spread, label = "Spread (NZDUSD - AUDUSD)")
    plt.axhline(spread.mean(), color='black')
    plt.legend()
    plt.title("Spread between NZDUSD and AUDUSD")
    plt.show()
    
    # Plot z-scores of the spread between NZDUSD and AUDUSD
    plt.figure(figsize=(8, 6), dpi=200)
    z_score = (spread - spread.mean())/spread.std()
    plt.plot(z_score, label = "Z Scores")
    plt.axhline(z_score.mean(), color = 'black')
    plt.axhline(2.0, color='red') 
    plt.axhline(-2.0, color='green') 
    plt.legend(loc = 'best')
    plt.title("Z-scores for Spread between NZDUSD and AUDUSD")
    plt.show()

if __name__ == "__main__":
    get_pairs_info()
    audnzd_info()
    audnzd_strategy()
    