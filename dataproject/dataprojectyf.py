# packages for data analysis
import pandas as pd
import numpy as np
import re
import warnings

# packages for data visualization
from IPython.display import display

# packages for data collection
import requests
import yfinance as yf


warnings.simplefilter(action='ignore', category=FutureWarning)

def yf_api(ticker, start, end):


    df = yf.download(ticker, start=start, end=end, interval="1d")
    return df

def fetch_data(mode="house", print_df = False):
    """
    Fetches data from the House or Senate Stockwatcher API.

    arguments:
        mode: string, either "house" or "senate"
        print_df: boolean, whether to print the dataframe

    returns:
        df: pandas dataframe, containing the data
    """
    
    # a. set the url based on the mode
    if mode == "house":
        url = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
    elif mode == "senate":
        url = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
    else:
        print("invalid mode")
        return False
    
    # b. make the request
    response = requests.get(url)

    # c. check the response
    if response.status_code != 200:
        print("request failed")
        return False
    else:
        print("request successful")

    # d. parse the response
    data_json = response.json()
    data = [] # create an empty list to hold cleaned data


    for transaction in data_json: # loop through each transaction in the original data
        # i. extract relevant fields
        date = transaction['transaction_date']
        ticker = transaction['ticker']
        amount = transaction['amount']
        action = transaction['type']
        representative = transaction['representative']
        description = transaction['asset_description']
        party = transaction['party']

        # ii. append cleaned data to the list
        data.append({'date': date, 
                     'ticker': ticker, 
                     'amount': amount, 
                     'action': action, 
                     'representative': representative,
                     'party': party, 
                     'description': description})
    df = pd.DataFrame(data)

    # e. print the head of the dataframe
    if print_df:
        display(df)

    return df

def clean_data(df, print_df = False):
    """
    Cleans the data from the House or Senate Stockwatcher API.

    arguments:
        df: pandas dataframe, containing the data
        print_df: boolean, whether to print the dataframe

    returns:
        df: pandas dataframe, containing the cleaned data
    """
    # a. copies the dataframe to avoid modifying the original
    df = df.copy()

    # b. replaces empty strings with NaN
    df.replace('', np.nan, inplace=True)

    # c. cleans amount column
    df[['min_amount', 'max_amount']] = df['amount'].str.split('-', expand=True) # i. splits amount into two columns: min_amount and max_amount and drops the original column
    df.drop(columns=['amount'], inplace=True)

    for x in ['min_amount', 'max_amount']:  # ii. cleans min_amount and max_amount columns 
        df[x] = df[x].str.replace('$', '')
        df[x] = df[x].str.replace(',', '')
        df[x] = df[x].str.replace('+', '')
        df[x].replace('', np.nan, inplace=True)
        df[x] = df[x].astype('float')
    
    # d. cleans date column
    df.date = pd.to_datetime(df.date, errors='coerce') # convert to datetime
    
    print(df.date.isna().sum(), "invalid dates dropped")
    df = df[df.date.notna()] # drop rows with invalid dates

    # e. cleans ticker 
    df.ticker = df.ticker.str.upper() # convert to uppercase
    df.ticker = df.ticker.str.strip() # remove leading and trailing whitespace
    
    print(df.ticker.count() - df.ticker.str.isalnum().sum(), "invalid tickers dropped") # print number of invalid tickers dropped
    df = df[df.ticker.str.isalnum()] # drop rows with invalid ticker
    df.ticker = df.ticker.astype('str') # convert to string
    df.ticker.replace('FB', 'META', inplace=True) # replace FB with META (Facebook changed their ticker from FB to META in 2022)

    # f. cleans action column
    df.action = df.action.str.lower() # convert to lowercase
    df.action = df.action.str.strip() # remove leading and trailing whitespace
    df.action = df.action.astype('str') # convert to string

    # g. cleans representative column
    df.representative = df.representative.str.title() # capitalize first letter
    df.representative = df.representative.str.strip() # remove leading and trailing whitespace
    df.representative = df.representative.astype('category') # convert to string

    # h. cleans party column
    df.party = df.party.str.lower() # convert to lowercase
    df.party = df.party.str.strip() # remove leading and trailing whitespace
    df.party = df.party.str[:3] # keep only the three letters
    df.party = df.party.astype('str') # convert to string

    # i. cleans description column by removing options trading
    df.description = df.description.astype('str') # convert to string
    print(df.description.str.contains('options', case=False).sum(), "options trades dropped") # print number of options trades dropped
    df.drop(df[df.description.str.contains('options', case=False)].index, inplace=True) # drop rows with options trading

    # j. rebase index
    df.reset_index(drop=True, inplace=True)

    # k. print the head of the dataframe
    if print_df:
        display(df)

    return df

def parse_no_shares(df, pattern = r"([$]?\d+\.?\,?\d+[K]?)", print_df = False): # still work in progress
    """
    Parses out the numbers in the description column.

    arguments:
        df: pandas dataframe, containing the data
        pattern: regular expression, the pattern to search for

    returns:
        df: pandas dataframe, containing the data with additional columns
    """

    # a. copies the dataframe to avoid modifying the original
    df = df.copy()

    # b. apply regular expression to column
    matches = df['description'].str.findall(pattern)

    # c. create new dataframe with separate columns for each match
    df_new = pd.DataFrame(matches.tolist(), index=df.index)

    # d. rename columns to match pattern
    df_new.columns = [f'desc_match{i+1}' for i in range(len(df_new.columns))]

    # e. concatenate new dataframe with original dataframe
    df = pd.concat([df, df_new], axis=1)

    # f. print the head of the dataframe
    if print_df:
        display(df)
    
    return df

def average_amount(df):
    """
    Calculates the average amount of money spent on a stock.

    arguments:
        df: pandas dataframe, containing the data

    returns:
        df: pandas dataframe, containing the data with additional columns
    """
    # a. copies the dataframe to avoid modifying the original
    df = df.copy()

    # b. calculate average amount
    df['avg_amount'] = df[['min_amount', 'max_amount']].mean(axis=1)

    return df

def select_rep(df, rep, print_df = False):
    """
    Selects a representative from the dataframe.

    arguments:
        df: pandas dataframe, containing the data
        rep: string, the name of the representative

    returns:
        df: pandas dataframe, containing the data for the selected representative
    """
    # a. copies the dataframe to avoid modifying the original
    df = df.copy()


    check = df.representative.str.contains(rep, case=False).sum()

    # b. select representative
    df = df[df.representative == rep]

    # c. rebase index
    df.reset_index(drop=True, inplace=True)

    # d. print the dataframe
    if print_df:
        display(df)

    return df
def get_stock_data(df):

    # a. find unique tickers, and max and min dates
    tickers = df.ticker.unique()

    # b. find min and max dates for each ticker
    min_dates = []
    max_dates = []
    for ticker in tickers:
        min_dates.append(df[df.ticker == ticker].date.min())
        max_dates.append(df[df.ticker == ticker].date.max())
    
    # c. change max date to today if action is not sell_full
    for i in range(len(max_dates)):
        if df[df.ticker == tickers[i]].action.str.contains('sell_full', case=False).sum() == 0:
            max_dates[i] = pd.to_datetime('today')

    # d. download stock data
    for i in range(len(tickers)):
        stock_df = yf.download(tickers[i], start=min_dates[i], end=max_dates[i], progress=False)
        stock_df['ticker'] = tickers[i]
        stock_df['date'] = stock_df.index
        stock_df.to_csv(f'{tickers[i]}.csv', index=False)
    

    return stock_df


