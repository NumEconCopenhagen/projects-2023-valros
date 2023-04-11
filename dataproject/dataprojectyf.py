# packages for data analysis
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# packages for data visualization
from IPython.display import display

# packages for data collection
import requests
import yfinance as yf


warnings.simplefilter(action='ignore', category=FutureWarning)

def yf_api(ticker, start, end):


    df = yf.download(ticker, start=start, end=end, interval="1d")
    return df

def fetch_data(print_df = False):
    """
    Fetches data from the House or Senate Stockwatcher API.

    arguments:
        mode: string, either "house" or "senate"
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

    # j. rebase index
    df.sort_values(by=['representative','ticker', 'date'], inplace=True)
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
    df['amount'] = df[['min_amount', 'max_amount']].mean(axis=1)

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

    # a. find unique tickers, and min date
    tickers = " ".join(df.ticker.unique())
    min_date = df.date.min()
    max_date = pd.to_datetime('today')

    # b. download stock data
    stock_df = yf.download(tickers, start=min_date, end=max_date, progress=True)

    # c. keep only adjusted close
    stock_df = stock_df['Adj Close']

    # d. print the dataframe
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

    # drop ticker if there is no purchase data
    df = df[df.ticker.isin(df[df.action == 'purchase'].ticker.unique())]

    # find first purchase date for each ticker
    first_purchase = df[df.action == 'purchase'].groupby('ticker').date.min().reset_index()

    # drop all rows before the first purchase date
    df = pd.merge(df, first_purchase, on='ticker', how='left')
    df = df[df.date_x >= df.date_y]
    df.drop(columns=['date_y'], inplace=True)
    df.rename(columns={'date_x':'date'}, inplace=True)

    df.sort_values(by=['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # c. print the dataframe
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


    # running a loop for each ticker
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

def daily_return(df):
    """
    Takes data frame and groups it by date to only show daily returns for entire portfolio
    """
    df = df.copy()
    df = df.groupby('date').agg({'weighted_return': 'sum'}).reset_index()

    # set date as index
    df.set_index('date', inplace=True)
    return df


def plot_return(df, include_sp500 = False):
    """
    Plots the cummulitive return of the portfolio
    """
    df = df.copy()

    df['cum_return'] = (1 + df['weighted_return']).cumprod()-1
    df['cum_return'].plot()

    # a. set the title
    plt.title('Cumulative Return of Portfolio')

    # b. set the x-axis label
    plt.xlabel('Date')

    # c. change y-axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # d. include S&P 500
    if include_sp500:
        min_date = df.index.min()
        max_date = df.index.max()
        sp500 = yf.download('^GSPC', start=min_date, end=max_date)
        sp500['cum_return'] = (1 + sp500['Adj Close'].pct_change()).cumprod()-1
        sp500['cum_return'].plot()

    plt.show()