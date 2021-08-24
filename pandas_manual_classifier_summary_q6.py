# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 6
Description of Problem (just a 1-2 line summary!): 

######### READ ME : Flow of the script #####################    
   
Ticker Used - TM , Duration used - 2015 to 2019 
Ticker Used - SPY , Duration used - 2015 to 2019 

Training Set - 2015, 2016 , 2017
Test Set - 2018,2019

Step 1 : Please note import, export obect is used for reading the dataframe storeed from previous script
Step 2 : For each ticker , Trube label is calcualted
Step 3 : Plot function is used to plot
    
"""
import os
import csv
import pandas as pd
import pickle
import seaborn as sns
sns.set_theme(style="whitegrid")
pd.options.mode.chained_assignment = None


ticker='TM'
# Second ticker
ticker2='SPY'
here =os.path.abspath( __file__ )
input_dir =os.path.abspath(os.path.join(here ,os. pardir ))
ticker_file = os.path.join(input_dir, ticker + '.csv')
ticker_file_spy = os.path.join(input_dir, ticker2 + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    """    Code for Question 6
    """
        
    
    ## Reloading the dataframes for TM and SPY ticker 
    
    ## For TM ticker the best W* is  W=4
    ## For SPY ticker the best W* is  W=2
        
    # Reload the file fo TM ticker
    file_tm_ticker = open('object_df_year45.obj', 'rb') 
    df_year45_reloaded = pickle.load(file_tm_ticker, encoding='utf-8')
    
    #Reload the file for SPY Ticker
    file_spy_ticker = open('object_df_spy_year45.obj', 'rb') 
    df_spy_year45_reloaded = pickle.load(file_spy_ticker, encoding='utf-8')
    
       
   ### For  TM Ticker
   
    ## $100 to start with to buy
    
    bucket=100
    df_year45_reloaded['Amount_W*']=''
    
    ##ticker the best W* is  W=4    

    for i in  range(len(df_year45_reloaded)-1):
    #for i in  range(100):    
        
        if df_year45_reloaded.loc[i,"W4"]== '+' :
            
            # Per Prof lecture , amount*(1+r) if retrun is positive then multiply by return
            bucket=bucket*(1+(df_year45_reloaded.loc[i,"Return"]))
            
            df_year45_reloaded.at[i,'Amount_W*']=bucket
            
        else:
            
            # Per Prof lecture , amount*(1+r) if retrun is negtaive then multiply by 1
            bucket=bucket*(1+0)
            #pass
            df_year45_reloaded.at[i,'Amount_W*']=bucket        
    
         
    if df_year45_reloaded.loc[len(df_year45_reloaded)-1,'W4'] == '-':
        df_year45_reloaded.at[len(df_year45_reloaded)-1,'Amount_W*']=df_year45_reloaded.loc[len(df_year45_reloaded)-2,'Amount_W*']
    else:
        df_year45_reloaded.at[len(df_year45_reloaded)-1,'Amount_W*']=df_year45_reloaded.loc[len(df_year45_reloaded)-2,'Amount_W*']*(1+df_year45_reloaded.loc[len(df_year45_reloaded)-2,'W4'])
    
    bucket=100
    df_year45_reloaded['Amount_Ensemble']=''
    
    ## For Ticker TM - Calculating Ensemble amount
    
    
    for i in  range(len(df_year45_reloaded)-1):
    
        if df_year45_reloaded.loc[i,"Ensemble"]== '+' :
            
            # Per Prof lecture , amount*(1+r) if retrun is positive then multiply by return
            bucket=bucket*(1+(df_year45_reloaded.loc[i,"Return"]))
            
            df_year45_reloaded.at[i,'Amount_Ensemble']=bucket
            #print('i',i,'bucket',bucket,(1+(df_year45_reloaded.loc[i+1,"Return"])))
        else:
            
            # Per Prof lecture , amount*(1+r) if retrun is negtaive then multiply by 1
            bucket=bucket*(1+0)
            pass
            df_year45_reloaded.at[i,'Amount_Ensemble']=bucket        

    
    if df_year45_reloaded.loc[len(df_year45_reloaded)-1,'Ensemble'] == '-':
        df_year45_reloaded.at[len(df_year45_reloaded)-1,'Amount_Ensemble']=df_year45_reloaded.loc[len(df_year45_reloaded)-2,'Amount_Ensemble']
    else:
        df_year45_reloaded.at[len(df_year45_reloaded)-1,'Amount_Ensemble']=df_year45_reloaded.loc[len(df_year45_reloaded)-2,'Amount_Ensemble']*(1+df_year45_reloaded.loc[len(df_year45_reloaded)-2,'Ensemble'])       
    
    ### Buy and Hold strategy
        
    # Buy stock for $100 for the Open Price for TM ticker 
    bucket=100      
    df_year45_reloaded['Amount_Buy_Hold']=''
    # Buying the shared for $100 for TM tiket on the open price
    sharestm=round(bucket/df_year45_reloaded.loc[0,"Open"],3)
    
    for i in  range(len(df_year45_reloaded)-1):
    
        # Setting the starting amout for day 1        
        if i==0:
            df_year45_reloaded.at[i,'Amount_Buy_Hold']=bucket
        else:
            df_year45_reloaded.at[i,'Amount_Buy_Hold']=sharestm*df_year45_reloaded.loc[i,"Close"]
    
    ## Amoumtt on the last day of Buy and Hold 
    
    df_year45_reloaded.at[len(df_year45_reloaded)-1,'Amount_Buy_Hold']=sharestm*df_year45_reloaded.loc[len(df_year45_reloaded)-1,"Close"]
    
    ##### Plot 
    
        
    df_year45_reloaded['Amount_W4'] = pd.to_numeric(df_year45_reloaded['Amount_W*'])
    df_year45_reloaded['Amount_Ensemble'] = pd.to_numeric(df_year45_reloaded['Amount_Ensemble'])
    df_year45_reloaded['Amount_Buy_Hold'] = pd.to_numeric(df_year45_reloaded['Amount_Buy_Hold'])       
    df_test_tm=df_year45_reloaded[['Date','Amount_W*','Amount_Ensemble','Amount_Buy_Hold']]
    
    ## PLot for TM 
    
    df_test_tm.plot(x='Date',y=['Amount_W*','Amount_Ensemble','Amount_Buy_Hold'],title="Plot for Ticker : TM",ylabel="Amount($)",xlabel="Date Range")

    print("\n")
    print("Quesion 5 Part 1- Obbservation for Ticker TM  ")
    print("\n")
    print("Observation1 :Prediction using Ensemble method gave the best return for portfolio")
    print("Observation2 : Buy and Hold is surprising giving the best result in the portfolio since although the results fluctuates the portfolio is still positive")
    print("Observation3 : Best W* method is loosing money in portfolio and still  performing better than ensemble")

##################################################

   ### For  SPY Ticker
   
    ## $100 to start with to buy
    
    bucket=100
    df_spy_year45_reloaded['Amount_W*']=''
    
    ##ticker the best W is  W=4    

    for i in  range(len(df_spy_year45_reloaded)-1):
    #for i in  range(100):    
        
        if df_spy_year45_reloaded.loc[i,"W2"]== '+' :
            
            # Per Prof lecture , amount*(1+r) if retrun is positive then multiply by return
            bucket=bucket*(1+(df_spy_year45_reloaded.loc[i,"Return"]))
            
            df_spy_year45_reloaded.at[i,'Amount_W*']=bucket
            
        else:
            
            # Per Prof lecture , amount*(1+r) if retrun is negtaive then multiply by 1
            bucket=bucket*(1+0)
            #pass
            df_spy_year45_reloaded.at[i,'Amount_W*']=bucket        
    
         
    if df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-1,'W4'] == '-':
        df_spy_year45_reloaded.at[len(df_spy_year45_reloaded)-1,'Amount_W*']=df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'Amount_W*']
    else:
        df_spy_year45_reloaded.at[len(df_spy_year45_reloaded)-1,'Amount_W*']=df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'Amount_W*']*(1+df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'W4'])
    
    bucket=100
    df_spy_year45_reloaded['Amount_Ensemble']=''
    
    ## For Ticker TM - Calculating Ensemble amount
    
    
    for i in  range(len(df_spy_year45_reloaded)-1):
    
        if df_spy_year45_reloaded.loc[i,"Ensemble"]== '+' :
            
            # Per Prof lecture , amount*(1+r) if retrun is positive then multiply by return
            bucket=bucket*(1+(df_spy_year45_reloaded.loc[i,"Return"]))
            
            df_spy_year45_reloaded.at[i,'Amount_Ensemble']=bucket
            #print('i',i,'bucket',bucket,(1+(df_spy_year45_reloaded.loc[i+1,"Return"])))
        else:
            
            # Per Prof lecture , amount*(1+r) if retrun is negtaive then multiply by 1
            bucket=bucket*(1+0)
            pass
            df_spy_year45_reloaded.at[i,'Amount_Ensemble']=bucket        

    
    if df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-1,'Ensemble'] == '-':
        df_spy_year45_reloaded.at[len(df_spy_year45_reloaded)-1,'Amount_Ensemble']=df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'Amount_Ensemble']
    else:
        df_spy_year45_reloaded.at[len(df_spy_year45_reloaded)-1,'Amount_Ensemble']=df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'Amount_Ensemble']*(1+df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-2,'Ensemble'])       
    
    ### Buy and Hold strategy
        
    # Buy stock for $100 for the Open Price for TM ticker 
    bucket=100      
    df_spy_year45_reloaded['Amount_Buy_Hold']=''
    # Buying the shared for $100 for TM tiket on the open price
    sharestm=round(bucket/df_spy_year45_reloaded.loc[0,"Open"],3)
    
    for i in  range(len(df_spy_year45_reloaded)-1):
    
        # Setting the starting amout for day 1        
        if i==0:
            df_spy_year45_reloaded.at[i,'Amount_Buy_Hold']=bucket
        else:
            df_spy_year45_reloaded.at[i,'Amount_Buy_Hold']=sharestm*df_spy_year45_reloaded.loc[i,"Close"]
    
    ## Amoumtt on the last day of Buy and Hold 
    
    df_spy_year45_reloaded.at[len(df_spy_year45_reloaded)-1,'Amount_Buy_Hold']=sharestm*df_spy_year45_reloaded.loc[len(df_spy_year45_reloaded)-1,"Close"]
    
    ##### Plot 
    
        
    df_spy_year45_reloaded['Amount_W4'] = pd.to_numeric(df_spy_year45_reloaded['Amount_W*'])
    df_spy_year45_reloaded['Amount_Ensemble'] = pd.to_numeric(df_spy_year45_reloaded['Amount_Ensemble'])
    df_spy_year45_reloaded['Amount_Buy_Hold'] = pd.to_numeric(df_spy_year45_reloaded['Amount_Buy_Hold'])       
	
    df_test_spy=df_spy_year45_reloaded[['Date','Amount_W*','Amount_Ensemble','Amount_Buy_Hold']]
   
    
    df_test_spy.plot(x='Date',y=['Amount_W*','Amount_Ensemble','Amount_Buy_Hold'],title="Plot for Ticker : SPY",ylabel="Amount($)",xlabel="Date Range")
    
    print("\n")
    print("Quesion 5 - Obbservation for Ticker SPY - ")
    print("\n")
    print("Obbservation1 : Prediction using Enseble methond gave the best retrun for portfolio")
    print("Obbservation2 : Retrun using the Best W* although is greater than Buy and HOld still lower than the ensable method ")
    print("Obbservation3 : Retrun using the Buy and HOld method is lower than the other two methods ")
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)