Generating Technical Indicators:
===================
BitCoin Price data is downloaded using yahoofinance library
the code for technical indicators is in functions.py 
First Run mainCode.py in jupyter notebook using 
%run mainCode 

It creates the file BTC_Indicators.csv





BTCPriceFeatureEngineering.ipynb:
===============================
data is read from BTCPrice.csv
removed all the duplicates - here for duplicates we will just drop one and keep=First 
and generate BTCPriceWithoutDuplicates.csv
merge with BTC_Indicators.csv and generate BTCPriceMergedIndicatorsAndWithoutDuplicates
dropped the first 54 rows so that there are no null values in any feature and generated BTCPriceCleaned.csv
