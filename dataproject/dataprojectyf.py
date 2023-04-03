import yfinance as yf
import pandas as pd

def yf_api(ticker, start, end):


    df = yf.download(ticker, start=start, end=end, interval="1d")
    return df