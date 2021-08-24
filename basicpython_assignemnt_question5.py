# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 5
Description of Problem (just a 1-2 line summary!): Buy and Hold strategy

######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 per given in pdf 

Step 1 : CSV file are read and casted to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , seperate master list is created , all further logic is derived off this master list 
Step 3 : Approach take to buy is to Buy stock on first day of 2015 for $100 for the Open Price, this is the first day record 
Step 3a: NO of stock bought = $100/Open Price(on first day of 2015)
Step 4 : THE number stock bought on the first day is holded and sold on the last day of 2019 at CLose prise 
Step 4a: Final Amount = NO of stock bought*Close Price( on last day of 2019) 


    
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
    """    Code for Question 5 
    """
        
    
    # using uft encoding 
    # Casting the line of xls to a list 
    data= open(ticker_file,encoding='utf-8')
    csv_data = csv.reader(data, delimiter=',')
    data_line= list(csv_data)
        
    ## Reading the SPY ticker
    data_spy= open(ticker_file_spy,encoding='utf-8')
    csv_data_spy = csv.reader(data_spy, delimiter=',')
    data_line_spy= list(csv_data_spy)
    
    # creating empty list to segregate by years for SPY Ticker
    
   
    data_line_all_spy=[]
    
    
    for line in  data_line_spy[1:]:
        
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all_spy.append(line)
        else:
            pass           
    
    
    # First Day
    # Buy stock for $100 for the Open Price for SKY ticker 
    bucket=100
    
    shares=round(bucket/float(data_line_all_spy[0][7]),5)
    
    # creating empty list to segregate by years for TM Ticker
    
   
    data_line_all=[]
    
    
    for line in  data_line[1:]:
        
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all.append(line)
        else:
            pass           
    
    
    # First Day
    # Buy stock for $100 for the Open Price for TM ticker 
    bucket=100
    
    # Buying the shared for $100 for TM tiket on the open price
    sharestm=round(bucket/float(data_line_all[0][7]),5)    
    
    
    print('\n')
    print('******Qustion 5 *********')
    print('\n')
    print('For ticker',ticker2)
    print('\n')
    print("On first day of",data_line_all_spy[0][1],"on",data_line_all_spy[0][0], "the total of",round(shares,4),"stock where brought for $100","for the price of ticker at ",ticker2, "$",float(data_line_all_spy[0][7]))
    print('\n')
    print("On last day of",data_line_all_spy[-1][1],"on",data_line_all_spy[-1][0],"the stock of ticker",ticker2,"was sold for close price of $",data_line_all_spy[-1][10],"so total worth is $",round(float(data_line_all_spy[-1][10])*shares,4))
    
    print('\n')
    print('For ticker',ticker)
    print('\n')
    print("On first day of",data_line_all[0][1],"on",data_line_all[0][0], "the total of",round(shares,4),"stock where brought for $100","for the price of ticker at ",ticker, "$",float(data_line_all[0][7]))
    print('\n')
    print("On last day of",data_line_all[-1][1],"on",data_line_all[-1][0],"the stock of ticker",ticker,"was sold for close price of $",data_line_all_spy[-1][10],"so total worth is $",round(float(data_line_all[-1][10])*sharestm,4))
    
    print('\n')
    print('Question 5 - Part 2 - how do these results compare with results obtained in question 4?')
    print('Answer -Buy and hold strategy does not yield that much profit as you are one doing one transaction and do not necessarily time the market to good days etc')
    print('\n')
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)