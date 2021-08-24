# -*- coding: utf-8 -*-
"""
Name Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 4
Description of Problem (just a 1-2 line summary!): Following oraclw advise
    
######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 per given in pdf 

Step 1 : CSV file are read and casted to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , seperate master list is created , all further logic is derived off this master list 
Step 3:  The approach taken is to take Amount*(1+Return), take only positive retrun while skip the negative returns
Step 3 : The logic goes through a for loop and exmaines the return for next to day to decide on the calcuation 
Step 4 : The aggregated simple computing result is stored in bucket variable which holds the final amount

    
"""
import os
import csv


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
    """    Code for Question 4
    """
        
    #data= open('C:\\Users\\gaura\\Downloads\\CS677\\Mod1\\metcs677_Assignment1.1\\week_1_homework\\MSFT.csv',encoding='utf-8')
    # using uft encoding 
    # Casting the line of xls to a list 
    data= open(ticker_file,encoding='utf-8')
    csv_data = csv.reader(data, delimiter=',')
    data_line= list(csv_data)
        
    ## Reading the SPY ticker
    data_spy= open(ticker_file_spy,encoding='utf-8')
    csv_data_spy = csv.reader(data_spy, delimiter=',')
    data_line_spy= list(csv_data_spy)
    
    
    # creating empty list to segregate by years for TM Ticker from 2015 to 2019
    # makeing a list for 5 yrs from 2015 to 2019 
    
    data_line_all=[]
        
    
    for line in  data_line[1:]:
        #print(line[1])
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all.append(line)
        else:
            pass
    
    
    ## The following is the assumption
    ## Oracle will tell us if the retrun is postive then hold on to it and if it is negative then skip those days
    ## The approach taken is to take Amount*(1+Return), take only positive retrun while skip the negative returns
    ##This logic was take up during the Professor class
    ##
    ## $100 to start with to buy
    
    
    bucket=100

    # Loop till before the last day since the preductin for last day if of the following day
    
    for i,line in  enumerate(data_line_all[:-1]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all[i+1][13]) > 0 :
                        
            #bucket=round((((bucket)*(1+round((float(data_line_all[i+1][13])),3)))),3)
            bucket=bucket*(1+float(data_line_all[i+1][13]))
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
        
    print('\n')       
    print("Question 4 : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker,"is $",round(bucket,3))    
    
    
    # creating empty list to segregate by years for SPY Ticker

    data_line_all_spy=[]
    
    for line in  data_line_spy[1:]:
        #print(line[1])
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all_spy.append(line)
        else:
            pass           

    ## $100 to start with to buy
    
    bucket=100
    
    for i,line in  enumerate(data_line_all_spy[:-1]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all_spy[i+1][13]) > 0 :
                        
            
            bucket=bucket*(1+float(data_line_all_spy[i+1][13]))
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass        
        
    print('\n')
    print("Question 4 : money you have wih $100 start on 2015 to last trading day of 2019 for ticker",ticker2,"is $",round(bucket,3))    

    
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)