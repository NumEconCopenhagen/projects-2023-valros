# a. packages for data analysis
import pandas as pd
import numpy as np
import warnings
import sys

# b. packages for data visualization
from IPython.display import display
import ipywidgets as wg
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# c. packages for data collection
import requests
import yfinance as yf


# d. remove all FutureWarning, which are not relevant for this pro ject
warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_data(print_df = False):
    """
    Fetches data from the House Stockwatcher API.

    arguments:
        print_df: boolean, whether to print the dataframe

    returns:
        df: pandas dataframe, containing the data
    """
    
    # a. make the request
    response = requests.get("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json")

    # b. check the response
    if response.status_code != 200:
        print("request failed")
        return False
    else:
        print("request successful")

    # c. parse the response
    data_json = response.json()
    data = [] # create an empty list to hold cleaned data

    # d. loop through each transaction in the original data
    for transaction in data_json: 
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
    Cleans the data from the House Stockwatcher API.

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
    # i. splits amount into two columns: min_amount and max_amount and drops the original column
    df[['min_amount', 'max_amount']] = df['amount'].str.split('-', expand=True) 
    df.drop(columns=['amount'], inplace=True)
    
    # ii. cleans min_amount and max_amount columns
    for x in ['min_amount', 'max_amount']:   
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
    df.ticker = df.ticker.astype('string') # convert to string
    df.ticker.replace('FB', 'META', inplace=True) # replace FB with META (Facebook changed their ticker from FB to META in 2022)

    # f. cleans action column
    df.action = df.action.str.lower() # convert to lowercase
    df.action = df.action.str.strip() # remove leading and trailing whitespace
    df.action = df.action.astype('string') # convert to string

    # g. cleans representative column
    df.representative = df.representative.str.title() # capitalize first letter
    df.representative = df.representative.str.strip() # remove leading and trailing whitespace
    df.representative = df.representative.astype('category') # convert to string

    # h. cleans party column
    df.party = df.party.str.lower() # convert to lowercase
    df.party = df.party.str.strip() # remove leading and trailing whitespace
    df.party = df.party.str[:3] # keep only the three letters
    df.party = df.party.astype('string') # convert to string

    # i. cleans description column by removing options trading
    df.description = df.description.astype('string') # convert to string
    print(df.description.str.contains('options', case=False).sum(), "options trades dropped") # print number of options trades dropped
    df.drop(df[df.description.str.contains('options', case=False)].index, inplace=True) # drop rows with options trading
    
    # j. find all representatives that have made purchases
    print(df.representative.nunique()-df[df.action == 'purchase'].representative.nunique(), "representatives dropped because they have not made any purchases")
    reps = df[df.action == 'purchase'].representative.unique()
    df = df[df.representative.isin(reps)]

    # k. rebase index
    df.sort_values(by=['representative','ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # l. calculate average amount
    df['amount'] = df[['min_amount', 'max_amount']].mean(axis=1)

    # m. print the head of the dataframe
    if print_df:
        display(df)

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
    df.sort_values(by=['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # d. print the dataframe
    if print_df:
        display(df)

    return df

def get_stock_data(df, print_df = False):
    """
    Creates list of unique tickers and downloads stock data from Yahoo Finance.

    arguments:
        df: pandas dataframe, containing the data
        print_df: boolean, whether to print the dataframe

    returns:
        stock_df: pandas dataframe, containing the stock data
    """
    

    # a. find unique tickers, and min date
    tickers = " ".join(df.ticker.unique())
    min_date = df.date.min()
    max_date = pd.to_datetime('today')

    # b. download stock data
    print("Downloading stock data...")
    stock_df = yf.download(tickers, start=min_date, end=max_date, progress=True)

    # c. keep only adjusted close 
    stock_df = stock_df['Adj Close']

    # d. rename columns when only one ticker is selected and force to pd.DataFrame
    if len(df.ticker.unique()) == 1:
        stock_df = pd.DataFrame(stock_df)
        stock_df.rename(columns={'Adj Close':df.ticker.unique()[0]}, inplace=True)

    # e. print the dataframe
    if print_df:
        display(stock_df)

    return stock_df

def merge_data(df,stock, print_df = False):
    """
    Merge the stock data with the representative data.

    arguments:
        df: pandas dataframe, containing the represnetative data
        stock: pandas dataframe, containing the stock data

    returns:
        df: pandas dataframe, containing the merged data
    
    """
    # a. copies the dataframe to avoid modifying the original
    df = df.copy()
    stock = stock.copy()

    # b. merge dataframes
    df = pd.merge(stock.stack().reset_index(), df, left_on=['Date', 'level_1'], right_on=['date', 'ticker'], how='left')
    df.drop(columns=['date', 'ticker', 'representative', 'party', 'description', 'min_amount', 'max_amount'], inplace=True)
    df.rename(columns={'Date':'date','level_1': 'ticker', 0: 'price'}, inplace=True)
    df.ticker = df.ticker.astype('string')

    # d. setting <NA> values to 0 in the action column
    df.action.fillna('none', inplace=True)

    # e. drop ticker if there is no purchase data
    df = df[df.ticker.isin(df[df.action == 'purchase'].ticker.unique())]

    # f. find first purchase date for each ticker
    first_purchase = df[df.action == 'purchase'].groupby('ticker').date.min().reset_index()

    # g. drop all rows before the first purchase date
    df = pd.merge(df, first_purchase, on='ticker', how='left')
    df = df[df.date_x >= df.date_y]
    df.drop(columns=['date_y'], inplace=True)
    df.rename(columns={'date_x':'date'}, inplace=True)

    # h. rebase index
    df.sort_values(by=['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # i. print the dataframe
    if print_df:
        display(df)
    
    return df

def portfolio(df, print_df = False):
    """
    Calculates the number of shares in the portfolio.

    arguments
        df: pandas dataframe, containing the data

    returns:
        df: pandas dataframe, containing the data with additional columns
    """

    # a. copies the dataframe to avoid modifying the original
    df = df.copy()

    # b. creating new columns populated with 0 and empty dataframe to store results
    df['share_change'] = 0
    df['shares_owned'] = 0
    results_df = pd.DataFrame()


    # c. running a loop for each ticker
    tickers = df.ticker.unique()
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].reset_index()
        for i in range(len(ticker_df)):
            if i == 0:
                ticker_df.loc[i, 'share_change'] = ticker_df.loc[i, 'amount'] // ticker_df.loc[i, 'price']
                ticker_df.loc[i, 'shares_owned'] = ticker_df.loc[i, 'share_change']
            else:
                if ticker_df.loc[i, 'action'] == 'purchase':
                    ticker_df.loc[i, 'share_change'] = ticker_df.loc[i, 'amount'] // df.loc[i, 'price']
                elif ticker_df.loc[i, 'action'] == 'sale_partial':
                    ticker_df.loc[i, 'share_change'] = - (ticker_df.loc[i-1, 'shares_owned'] // 2)
                elif ticker_df.loc[i, 'action'] == 'sale_full':
                    ticker_df.loc[i, 'share_change'] = - ticker_df.loc[i-1, 'shares_owned']
                ticker_df.loc[i, 'shares_owned'] = ticker_df.loc[i-1, 'shares_owned'] + ticker_df.loc[i, 'share_change'] # should give 0
        results_df = results_df.append(ticker_df)
        
    results_df = results_df.reset_index(drop=True)
    results_df = results_df.drop(columns=['index'])

    # d. return of the shares
    results_df['daily_return'] = results_df.groupby('ticker')['price'].pct_change()

    # e. calculating portfolio weight for each ticker by value of the shares owned for each day
    results_df['value_ticker'] = results_df['price'] * results_df['shares_owned']

    # f. calculating portfolio value for each day
    results_df['value_portfolio'] = results_df.groupby('date')['value_ticker'].transform('sum')

    # g. calculating portfolio weight for each ticker by value of the shares owned for each day
    results_df['weight_ticker'] = results_df['value_ticker'] / results_df['value_portfolio']

    # h. calculating portfolio return for each day
    results_df['weighted_return'] = results_df['daily_return'] * results_df['weight_ticker']

    # j. print the dataframe
    if print_df:
        display(results_df)

    return results_df

def daily_return(df, print_df = False):
    """
    Takes data frame and groups it by date to only show daily returns for entire portfolio
    """
    df = df.copy()
    df = df.groupby('date').agg({'weighted_return': 'sum'}).reset_index()

    # set date as index
    df.set_index('date', inplace=True)

    # print the dataframe
    if print_df:
        display(df)
    return df

def plot_return(df, include_sp500 = False, title = 'Nancy Pelosi'):
    """
    Plots the cummulitive return of the portfolio
    """
    df = df.copy()

    df['cum_return'] = (1 + df['weighted_return']).cumprod()-1
    df['cum_return'].plot()

    # a. set the title
    plt.title('Cumulative Return of ' + title + "'s Portfolio")

    # b. set the x-axis label
    plt.xlabel('Date')
    
    # c. change y-axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # d. include S&P 500
    if include_sp500:
        min_date = df.index.min()
        max_date = df.index.max()
        print("Downloading S&P 500 data...")
        sp500 = yf.download('^GSPC', start=min_date, end=max_date)
        sp500['cum_return'] = (1 + sp500['Adj Close'].pct_change()).cumprod()-1
        sp500['cum_return'].plot()
        plt.legend([f'{title} Portfolio', 'S&P 500'])

    plt.show()

def widget(df,name):
    """
    Calls all the functions to get a plot of the portfolio return and the S&P 500

    arguments
        df: pandas dataframe, cleaned house dataset
        name: string, name of the representative

    returns:
        plot of the portfolio return and the S&P 500
    """

    # a. stops process if "Click here" is selected
    if name == 'Click here':
        print('Please select a representative')
        return None

    # b. copies the dataframe to avoid modifying the original
    df = df.copy()

    # c. select the representative
    df = select_rep(df, name)

    # d. get stock data
    stock = get_stock_data(df)

    # e. stops the function if no stock data is found
    if stock.empty:
        print('No stock data found')
        return None

    # f. merge the dataframes
    merge = merge_data(df,stock)

    # g. calculate portfolio and weighted return
    portfolio_data = portfolio(merge)

    # h. calculate daily return on entire portfolio
    portfolio_return = daily_return(portfolio_data)

    # i. plot the return
    plot_return(portfolio_return, include_sp500 = True, title = name)
