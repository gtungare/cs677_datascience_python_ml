# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 1
Description of Problem (just a 1-2 line summary!): Quesion 1


######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 
Ticker Used - SPY , Duration used - 2015 to 2019 

Training Set - 2015, 2016 , 2017
Test Set - 2018,2019

Step 1 : CSV file are read and created to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , Trube label is calcualted
Step 2 : For each ticker ,For loop is used to calculate the probability

Please note 
   
"""
import os
import csv
import pandas as pd
import numpy as np

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
    """    Code for Question 1
    """
        
    
    # using uft encoding 
    # Casting the line of xls to a list 
    data= open(ticker_file,encoding='utf-8')
    csv_data = csv.reader(data, delimiter=',')
    
    data_line= list(csv_data)
    data_tickertm = pd.DataFrame(data_line)
    
    # Reading using pandas
    df=pd.read_csv(ticker_file) 
        
    #data_tickertm['TrueLabel'] = np.where(float(data_tickertm[13]) > 0,'+','-')
    
    ## Reading the SPY ticker
    data_spy= open(ticker_file_spy,encoding='utf-8')
    csv_data_spy = csv.reader(data_spy, delimiter=',')
    data_line_spy= list(csv_data_spy)
    
    # Reading using pandas
    df_spy=pd.read_csv(ticker_file_spy) 
    
    ## Creating a new column called TrueLabel 
    ## If the return is greater than 0 then "Plus" else "Neg"
    
    df['TrueLabel'] = np.where(df["Return"] > 0,'+','-')
    
    # Splitting between test and train set for TM Ticker
    
    df_year123 = df[df["Year"].isin([2015,2016,2017])]
    df_year45 = df[df["Year"].isin([2018,2019])]

    ## Creating a new column called TrueLabel for SPY ticker
    ## If the return is greater than 0 then "Plus" else "Neg"
    
    df_spy['TrueLabel'] = np.where(df_spy["Return"] > 0,'+','-')
    
    # Splitting between test and train set for TM Ticker
    
    df_spy_year123 = df_spy[df_spy["Year"].isin([2015,2016,2017])]
    df_spy_year45 = df_spy[df_spy["Year"].isin([2018,2019])]
    
    df_year45.reset_index(drop=True,inplace=True)
    df_spy_year45.reset_index(drop=True,inplace=True)
    
    ##### Part 1
    
    print("\n")
    print("Question 1 - Part 1 - For Ticker TM" )
    print(df_year45[["Return","TrueLabel"]].head(10))
    print("\n")
    print("Question 1 - Part 1 - For Ticker SPY" )
    print(df_spy_year45[["Return","TrueLabel"]].head(10))
    
    ###############
    
    pos_tm=df_year123[df_year123["TrueLabel"]=='+']["TrueLabel"].count()
    total_tm=df_year123["TrueLabel"].count()
    
    pos_spy=df_spy_year123[df_spy_year123["TrueLabel"]=='+']["TrueLabel"].count()
    total_spy=df_spy_year123["TrueLabel"].count()
    
    # Probability p(A)=N(A)/N, where A is an event, N(A) is the number of times the event A occurs, and N is the total number of times the experiment is performed
    
    
        
    print("\n")
    print("Question 1 - Part 2 The default probability p for ticker",ticker," that the next day is a UP day is",round(pos_tm/total_tm,3))
    
    print("\n")
    print("Question 1 - Part 2 The default probability p for ticker",ticker2," that the next day is a UP day is",round(pos_spy/total_spy,3))
    
    ## Calculation for Question 1 part 3
    
    ## for TM
    
    # for K=1
    k=1
    # using the first "down" days followed by an "up" day
    s='-+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=1 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-2)):
        
        #if 1==1:
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]==s:
            c = c+1            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker,": the probability for K=",k," for [- +] is equal to ",round( c/total_tm,3))    
    
    ## for SPY
    # for K=1
    k=1
    # using the first "down" days followed by an "up" day
    s='-+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=1 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-2)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker2,":  the probability for K=",k," for [- +] is equal to ",round( c/total_spy,3))    
    
    ## for ticer TM
    # for K=2
    k=2
    # using for [- - +] 
    s='--+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-3)):
        
        #if 1==1:
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]==s:
            c = c+1
            #print('cheking')
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker," : the probability for K=",k," for [- - +] is equal to ",round( c/total_tm,3))    
    
    ## for ticer SPY
    # for K=2
    k=2
    # Using [- - +] 
    s='--+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-3)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]==s:
            c = c+1
            #print('cheking')
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker2," : the probability for K=",k," for [- - +] is equal to ",round( c/total_spy,3))    
    
    # For Ticker TM
    # for K=3
    k=3
    # using the first "down" days followed by an "up" day
    s='---+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-4)):
        
        #if 1==1:
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]==s:
            c = c+1
            #print('cheking')
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker,": the probability for K=",k," for [- - - +] is equal to ",round( c/total_tm,3))    
    
    # For Ticker SPY
    # for K=3
    k=3
    # using  [- - - +] 
    s='---+'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-4)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 3 -For Ticker",ticker2," : the probability for K=",k," for [- - - +] is equal to ",round( c/total_spy,3))    
    
    ########### question 1 part 4
    
    ## Calculation for Question 1 part 4
    
    ## for TM Ticker
    
    
    # for K=1
    k=1
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    s='++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=1 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-2)):
        
        
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]==s:
            c = c+1
            #print('cheking')
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker,"  the probability for K=",k," for [+ +] is equal to ",round( c/total_tm,3))    
    
    ## for SPY Ticker
    # for K=1
    k=1
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    ##for [+ +] 
    s='++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=1 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-2)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker2," : the probability for K=",k," for [+ +] is equal to ",round( c/total_spy,3))    
    
    
    ## For TM Ticker 
    # for K=2
    k=2
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    ## for [+ + + ] 
    s='+++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=2 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-3)):
        
        #if 1==1:
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker," : the probability for K=",k," for [+ + +] is equal to ",round( c/total_tm,3))    
    
    
    ## For SPY Ticker 
    # for K=2
    k=2
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    s='+++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=2 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-3)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker2,"  the probability for K=",k," for [+ + +] is equal to ",round( c/total_spy,3))    
    
    # For TM Ticker
    # for K=3
    k=3
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    ## using  [+ + +  +] 
    
    s='++++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_year123)-4)):
        
        #if 1==1:
        if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]==s:
            c = c+1
            #print('cheking')
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker," : the probability for K=",k," for [+ + + +] is equal to ",round( c/total_tm,3))    
    
    # For SPY Ticker
    # for K=3
    k=3
    # probability that after seeing k consecutive "up days", the next day is still an "up day"
    s='++++'
    c=0
    
    ## Index is reset to start from 0 
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    # since k=3 the list goes till the second last row in the training set 
    
    for i in range((len(df_spy_year123)-4)):
        
        #if 1==1:
        if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]==s:
            c = c+1
            
        else:
            pass    
    
    
    print("\n")
    print("Question 1 - Part 4 -For Ticker",ticker2," : the probability for K=",k," for [+ + + +] is equal to ",round( c/total_spy,3))    
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)