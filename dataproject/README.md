# Data analysis project

Our project is titled **How good are house representatives when investing?** and investigates house representatives performance in the financial market. Congress members are legally required to declare any transaction they or a family member performs no later than 45 days after the transaction has taken place. This data is publicly available and in this data project we want to see if the house representatives outperform the market, which could indicate that they are in violation of the STOCK (Stop Trading on Congressional Knowledge) Act from 2012.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb) which uses functions defined in [dataproject.py](dataproject.py)

We use the **following data**:

1. Data on House Representatives' financial transactions from [House Stockwatcher](housestockwatcher.com)
2. Stock data from [Yahoo Finance](https://finance.yahoo.com/)

Both datasets are pulled directly from their respective API's when running the code

**Dependencies:** Apart from a standard Anaconda Python 3 installation and the ``.py``-file, the project requires the following installations:
- ``yfinance`` can be installed by running ``pip install yfinance``
- ``request`` can be installed by running ``pip install requests``
