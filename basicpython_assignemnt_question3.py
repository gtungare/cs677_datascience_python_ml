# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 3
Description of Problem (just a 1-2 line summary!): Question 3 Compute the aggregate table across all 5 years,
one table for both your stock and one table for S&P-500
    
######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 per given in pdf 

Step 1 : CSV file are read and casted to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , seperate master list is created , all further logic is derived off this master list 
Step 3 : For each ticker , for each day Monday...to Friday,, again seperate sublist is created further
Step 4 : Within each sublist further sublist is created for every attribute asked in the question 
Step 5 : For calcuation statistics package is used for calculation of mean and standard deviation
Step 6 : For printing output table "PrettyTable" is used for easy readibility
         
         Please expand the console to see the output properly   
    
"""
import os
import csv
import statistics

# run this  ! pip install PrettyTable
from prettytable import PrettyTable

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
    """    Code for Question 3 
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
    
    
    # creating empty list to segregate by years for TM Ticker
    
    data_line_all=[]
    
    for line in  data_line[1:]:
        #print(line[1])
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all.append(line)
        else:
            pass
    
    # creating empty list to segregate by years for SPY Ticker
    
    data_line_all_spy=[]
    
    for line in  data_line_spy[1:]:
        #print(line[1])
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all_spy.append(line)
        else:
            pass           
    
# For year TM Ticker  Creating list for Daily return, Negative returns and pos return 
    
    monday_all_neg_return_tm_all=[]
    monday_all_pos_return_tm_all=[]
    monday_comb_return_tm_all=[]
    tuesday_all_neg_return_tm_all=[]
    tuesday_all_pos_return_tm_all=[]
    tuesday_comb_return_tm_all=[]
    wednesday_all_neg_return_tm_all=[]
    wednesday_all_pos_return_tm_all=[]
    wednesday_comb_return_tm_all=[]
    thursday_all_neg_return_tm_all=[]
    thursday_all_pos_return_tm_all=[]
    thursday_comb_return_tm_all=[]
    friday_all_neg_return_tm_all=[]
    friday_all_pos_return_tm_all=[]
    friday_comb_return_tm_all=[]

    for line in  data_line_all[:]:
        if line[4]== 'Monday' :
            monday_comb_return_tm_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_tm_all.append(float(line[-3]))
            else:
                monday_all_neg_return_tm_all.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_tm_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_tm_all.append(float(line[-3]))
            else:
                tuesday_all_neg_return_tm_all.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_tm_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_tm_all.append(float(line[-3]))
            else:
                wednesday_all_neg_return_tm_all.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_tm_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_tm_all.append(float(line[-3]))
            else:
                thursday_all_neg_return_tm_all.append(float(line[-3]))        
        else :
            friday_comb_return_tm_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_tm_all.append(float(line[-3]))
            else:
                friday_all_neg_return_tm_all.append(float(line[-3]))        
        
    ## Summarizing the result for TM Ticker - Monday
    monday_comb_tm_all=['Monday']
    
    monday_r_neg_tm_all_mean = round(statistics.mean(monday_all_neg_return_tm_all),5)
    monday_r_neg_tm_all_std = round(statistics.stdev(monday_all_neg_return_tm_all),5)
    monday_r_neg_tm_all_cnt = len(monday_all_neg_return_tm_all)
    monday_r_pos_tm_all_mean = round(statistics.mean(monday_all_pos_return_tm_all),5)
    monday_r_pos_tm_all_std = round(statistics.stdev(monday_all_pos_return_tm_all),5)
    monday_r_pos_tm_all_cnt = len(monday_all_pos_return_tm_all)
    monday_r_comb_tm_all_mean = round(statistics.mean(monday_comb_return_tm_all),5)
    monday_r_comb_tm_all_std = round(statistics.stdev(monday_comb_return_tm_all),5)
    
    # Appendig the result to empty string for Monday TM Ticker 
    monday_comb_tm_all.append(monday_r_neg_tm_all_mean)
    monday_comb_tm_all.append(monday_r_neg_tm_all_std)
    monday_comb_tm_all.append(monday_r_neg_tm_all_cnt)
    monday_comb_tm_all.append(monday_r_pos_tm_all_mean)
    monday_comb_tm_all.append(monday_r_pos_tm_all_std)
    monday_comb_tm_all.append(monday_r_pos_tm_all_cnt)
    monday_comb_tm_all.append(monday_r_comb_tm_all_mean)
    monday_comb_tm_all.append(monday_r_comb_tm_all_std)

    ## Summarizing the result for TM Ticker - tuesday
    tuesday_comb_tm_all=['Tuesday']
    
    tuesday_r_neg_tm_all_mean = round(statistics.mean(tuesday_all_neg_return_tm_all),5)
    tuesday_r_neg_tm_all_std = round(statistics.stdev(tuesday_all_neg_return_tm_all),5)
    tuesday_r_neg_tm_all_cnt = len(tuesday_all_neg_return_tm_all)
    tuesday_r_pos_tm_all_mean = round(statistics.mean(tuesday_all_pos_return_tm_all),5)
    tuesday_r_pos_tm_all_std = round(statistics.stdev(tuesday_all_pos_return_tm_all),5)
    tuesday_r_pos_tm_all_cnt = len(tuesday_all_pos_return_tm_all)
    tuesday_r_comb_tm_all_mean = round(statistics.mean(tuesday_comb_return_tm_all),5)
    tuesday_r_comb_tm_all_std = round(statistics.stdev(tuesday_comb_return_tm_all),5)
    
    # Appendig the result to empty string for tuesday TM Ticker 
    tuesday_comb_tm_all.append(tuesday_r_neg_tm_all_mean)
    tuesday_comb_tm_all.append(tuesday_r_neg_tm_all_std)
    tuesday_comb_tm_all.append(tuesday_r_neg_tm_all_cnt)
    tuesday_comb_tm_all.append(tuesday_r_pos_tm_all_mean)
    tuesday_comb_tm_all.append(tuesday_r_pos_tm_all_std)
    tuesday_comb_tm_all.append(tuesday_r_pos_tm_all_cnt)
    tuesday_comb_tm_all.append(tuesday_r_comb_tm_all_mean)
    tuesday_comb_tm_all.append(tuesday_r_comb_tm_all_std)    
    
    ## Summarizing the result for TM Ticker - wednesday
    wednesday_comb_tm_all=['Wednesday']
    
    wednesday_r_neg_tm_all_mean = round(statistics.mean(wednesday_all_neg_return_tm_all),5)
    wednesday_r_neg_tm_all_std = round(statistics.stdev(wednesday_all_neg_return_tm_all),5)
    wednesday_r_neg_tm_all_cnt = len(wednesday_all_neg_return_tm_all)
    wednesday_r_pos_tm_all_mean = round(statistics.mean(wednesday_all_pos_return_tm_all),5)
    wednesday_r_pos_tm_all_std = round(statistics.stdev(wednesday_all_pos_return_tm_all),5)
    wednesday_r_pos_tm_all_cnt = len(wednesday_all_pos_return_tm_all)
    wednesday_r_comb_tm_all_mean = round(statistics.mean(wednesday_comb_return_tm_all),5)
    wednesday_r_comb_tm_all_std = round(statistics.stdev(wednesday_comb_return_tm_all),5)
    
    # Appendig the result to empty string for wednesday TM Ticker 
    wednesday_comb_tm_all.append(wednesday_r_neg_tm_all_mean)
    wednesday_comb_tm_all.append(wednesday_r_neg_tm_all_std)
    wednesday_comb_tm_all.append(wednesday_r_neg_tm_all_cnt)
    wednesday_comb_tm_all.append(wednesday_r_pos_tm_all_mean)
    wednesday_comb_tm_all.append(wednesday_r_pos_tm_all_std)
    wednesday_comb_tm_all.append(wednesday_r_pos_tm_all_cnt)
    wednesday_comb_tm_all.append(wednesday_r_comb_tm_all_mean)
    wednesday_comb_tm_all.append(wednesday_r_comb_tm_all_std)
    
    ## Summarizing the result for TM Ticker - thursday
    thursday_comb_tm_all=['Thursday']
    
    thursday_r_neg_tm_all_mean = round(statistics.mean(thursday_all_neg_return_tm_all),5)
    thursday_r_neg_tm_all_std = round(statistics.stdev(thursday_all_neg_return_tm_all),5)
    thursday_r_neg_tm_all_cnt = len(thursday_all_neg_return_tm_all)
    thursday_r_pos_tm_all_mean = round(statistics.mean(thursday_all_pos_return_tm_all),5)
    thursday_r_pos_tm_all_std = round(statistics.stdev(thursday_all_pos_return_tm_all),5)
    thursday_r_pos_tm_all_cnt = len(thursday_all_pos_return_tm_all)
    thursday_r_comb_tm_all_mean = round(statistics.mean(thursday_comb_return_tm_all),5)
    thursday_r_comb_tm_all_std = round(statistics.stdev(thursday_comb_return_tm_all),5)
    
    # Appendig the result to empty string for thursday TM Ticker 
    thursday_comb_tm_all.append(thursday_r_neg_tm_all_mean)
    thursday_comb_tm_all.append(thursday_r_neg_tm_all_std)
    thursday_comb_tm_all.append(thursday_r_neg_tm_all_cnt)
    thursday_comb_tm_all.append(thursday_r_pos_tm_all_mean)
    thursday_comb_tm_all.append(thursday_r_pos_tm_all_std)
    thursday_comb_tm_all.append(thursday_r_pos_tm_all_cnt)
    thursday_comb_tm_all.append(thursday_r_comb_tm_all_mean)
    thursday_comb_tm_all.append(thursday_r_comb_tm_all_std)    

    ## Summarizing the result for TM Ticker - friday
    friday_comb_tm_all=['Friday']
    
    friday_r_neg_tm_all_mean = round(statistics.mean(friday_all_neg_return_tm_all),5)
    friday_r_neg_tm_all_std = round(statistics.stdev(friday_all_neg_return_tm_all),5)
    friday_r_neg_tm_all_cnt = len(friday_all_neg_return_tm_all)
    friday_r_pos_tm_all_mean = round(statistics.mean(friday_all_pos_return_tm_all),5)
    friday_r_pos_tm_all_std = round(statistics.stdev(friday_all_pos_return_tm_all),5)
    friday_r_pos_tm_all_cnt = len(friday_all_pos_return_tm_all)
    friday_r_comb_tm_all_mean = round(statistics.mean(friday_comb_return_tm_all),5)
    friday_r_comb_tm_all_std = round(statistics.stdev(friday_comb_return_tm_all),5)
    
    # Appendig the result to empty string for friday TM Ticker 
    friday_comb_tm_all.append(friday_r_neg_tm_all_mean)
    friday_comb_tm_all.append(friday_r_neg_tm_all_std)
    friday_comb_tm_all.append(friday_r_neg_tm_all_cnt)
    friday_comb_tm_all.append(friday_r_pos_tm_all_mean)
    friday_comb_tm_all.append(friday_r_pos_tm_all_std)
    friday_comb_tm_all.append(friday_r_pos_tm_all_cnt)
    friday_comb_tm_all.append(friday_r_comb_tm_all_mean)
    friday_comb_tm_all.append(friday_r_comb_tm_all_std)
    
    ## Printing the table for TM Ticker 2014 to 2019
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_tm_all)
    t.add_row(tuesday_comb_tm_all)
    t.add_row(wednesday_comb_tm_all)
    t.add_row(thursday_comb_tm_all)
    t.add_row(friday_comb_tm_all)
    
    print('\n')
    print('Year- 2014 to 2019 TM Ticker  \n')
    print(t)

# For year SPY Ticker  Creating list for Daily return, Negative returns and pos return 
    
    monday_all_neg_return_spy_all=[]
    monday_all_pos_return_spy_all=[]
    monday_comb_return_spy_all=[]
    tuesday_all_neg_return_spy_all=[]
    tuesday_all_pos_return_spy_all=[]
    tuesday_comb_return_spy_all=[]
    wednesday_all_neg_return_spy_all=[]
    wednesday_all_pos_return_spy_all=[]
    wednesday_comb_return_spy_all=[]
    thursday_all_neg_return_spy_all=[]
    thursday_all_pos_return_spy_all=[]
    thursday_comb_return_spy_all=[]
    friday_all_neg_return_spy_all=[]
    friday_all_pos_return_spy_all=[]
    friday_comb_return_spy_all=[]

    for line in  data_line_all_spy[:]:
        if line[4]== 'Monday' :
            monday_comb_return_spy_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_spy_all.append(float(line[-3]))
            else:
                monday_all_neg_return_spy_all.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_spy_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_spy_all.append(float(line[-3]))
            else:
                tuesday_all_neg_return_spy_all.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_spy_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_spy_all.append(float(line[-3]))
            else:
                wednesday_all_neg_return_spy_all.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_spy_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_spy_all.append(float(line[-3]))
            else:
                thursday_all_neg_return_spy_all.append(float(line[-3]))        
        else :
            friday_comb_return_spy_all.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_spy_all.append(float(line[-3]))
            else:
                friday_all_neg_return_spy_all.append(float(line[-3]))        
        
    ## Summarizing the result for SPY Ticker - Monday
    monday_comb_spy_all=['Monday']
    
    monday_r_neg_spy_all_mean = round(statistics.mean(monday_all_neg_return_spy_all),5)
    monday_r_neg_spy_all_std = round(statistics.stdev(monday_all_neg_return_spy_all),5)
    monday_r_neg_spy_all_cnt = len(monday_all_neg_return_spy_all)
    monday_r_pos_spy_all_mean = round(statistics.mean(monday_all_pos_return_spy_all),5)
    monday_r_pos_spy_all_std = round(statistics.stdev(monday_all_pos_return_spy_all),5)
    monday_r_pos_spy_all_cnt = len(monday_all_pos_return_spy_all)
    monday_r_comb_spy_all_mean = round(statistics.mean(monday_comb_return_spy_all),5)
    monday_r_comb_spy_all_std = round(statistics.stdev(monday_comb_return_spy_all),5)
    
    # Appendig the result to empty string for Monday SPY Ticker 
    monday_comb_spy_all.append(monday_r_neg_spy_all_mean)
    monday_comb_spy_all.append(monday_r_neg_spy_all_std)
    monday_comb_spy_all.append(monday_r_neg_spy_all_cnt)
    monday_comb_spy_all.append(monday_r_pos_spy_all_mean)
    monday_comb_spy_all.append(monday_r_pos_spy_all_std)
    monday_comb_spy_all.append(monday_r_pos_spy_all_cnt)
    monday_comb_spy_all.append(monday_r_comb_spy_all_mean)
    monday_comb_spy_all.append(monday_r_comb_spy_all_std)

    ## Summarizing the result for SPY Ticker - tuesday
    tuesday_comb_spy_all=['Tuesday']
    
    tuesday_r_neg_spy_all_mean = round(statistics.mean(tuesday_all_neg_return_spy_all),5)
    tuesday_r_neg_spy_all_std = round(statistics.stdev(tuesday_all_neg_return_spy_all),5)
    tuesday_r_neg_spy_all_cnt = len(tuesday_all_neg_return_spy_all)
    tuesday_r_pos_spy_all_mean = round(statistics.mean(tuesday_all_pos_return_spy_all),5)
    tuesday_r_pos_spy_all_std = round(statistics.stdev(tuesday_all_pos_return_spy_all),5)
    tuesday_r_pos_spy_all_cnt = len(tuesday_all_pos_return_spy_all)
    tuesday_r_comb_spy_all_mean = round(statistics.mean(tuesday_comb_return_spy_all),5)
    tuesday_r_comb_spy_all_std = round(statistics.stdev(tuesday_comb_return_spy_all),5)
    
    # Appendig the result to empty string for tuesday SPY Ticker 
    tuesday_comb_spy_all.append(tuesday_r_neg_spy_all_mean)
    tuesday_comb_spy_all.append(tuesday_r_neg_spy_all_std)
    tuesday_comb_spy_all.append(tuesday_r_neg_spy_all_cnt)
    tuesday_comb_spy_all.append(tuesday_r_pos_spy_all_mean)
    tuesday_comb_spy_all.append(tuesday_r_pos_spy_all_std)
    tuesday_comb_spy_all.append(tuesday_r_pos_spy_all_cnt)
    tuesday_comb_spy_all.append(tuesday_r_comb_spy_all_mean)
    tuesday_comb_spy_all.append(tuesday_r_comb_spy_all_std)    
    
    ## Summarizing the result for SPY Ticker - wednesday
    wednesday_comb_spy_all=['Wednesday']
    
    wednesday_r_neg_spy_all_mean = round(statistics.mean(wednesday_all_neg_return_spy_all),5)
    wednesday_r_neg_spy_all_std = round(statistics.stdev(wednesday_all_neg_return_spy_all),5)
    wednesday_r_neg_spy_all_cnt = len(wednesday_all_neg_return_spy_all)
    wednesday_r_pos_spy_all_mean = round(statistics.mean(wednesday_all_pos_return_spy_all),5)
    wednesday_r_pos_spy_all_std = round(statistics.stdev(wednesday_all_pos_return_spy_all),5)
    wednesday_r_pos_spy_all_cnt = len(wednesday_all_pos_return_spy_all)
    wednesday_r_comb_spy_all_mean = round(statistics.mean(wednesday_comb_return_spy_all),5)
    wednesday_r_comb_spy_all_std = round(statistics.stdev(wednesday_comb_return_spy_all),5)
    
    # Appendig the result to empty string for wednesday SPY Ticker 
    wednesday_comb_spy_all.append(wednesday_r_neg_spy_all_mean)
    wednesday_comb_spy_all.append(wednesday_r_neg_spy_all_std)
    wednesday_comb_spy_all.append(wednesday_r_neg_spy_all_cnt)
    wednesday_comb_spy_all.append(wednesday_r_pos_spy_all_mean)
    wednesday_comb_spy_all.append(wednesday_r_pos_spy_all_std)
    wednesday_comb_spy_all.append(wednesday_r_pos_spy_all_cnt)
    wednesday_comb_spy_all.append(wednesday_r_comb_spy_all_mean)
    wednesday_comb_spy_all.append(wednesday_r_comb_spy_all_std)
    
    ## Summarizing the result for SPY Ticker - thursday
    thursday_comb_spy_all=['Thursday']
    
    thursday_r_neg_spy_all_mean = round(statistics.mean(thursday_all_neg_return_spy_all),5)
    thursday_r_neg_spy_all_std = round(statistics.stdev(thursday_all_neg_return_spy_all),5)
    thursday_r_neg_spy_all_cnt = len(thursday_all_neg_return_spy_all)
    thursday_r_pos_spy_all_mean = round(statistics.mean(thursday_all_pos_return_spy_all),5)
    thursday_r_pos_spy_all_std = round(statistics.stdev(thursday_all_pos_return_spy_all),5)
    thursday_r_pos_spy_all_cnt = len(thursday_all_pos_return_spy_all)
    thursday_r_comb_spy_all_mean = round(statistics.mean(thursday_comb_return_spy_all),5)
    thursday_r_comb_spy_all_std = round(statistics.stdev(thursday_comb_return_spy_all),5)
    
    # Appendig the result to empty string for thursday SPY Ticker 
    thursday_comb_spy_all.append(thursday_r_neg_spy_all_mean)
    thursday_comb_spy_all.append(thursday_r_neg_spy_all_std)
    thursday_comb_spy_all.append(thursday_r_neg_spy_all_cnt)
    thursday_comb_spy_all.append(thursday_r_pos_spy_all_mean)
    thursday_comb_spy_all.append(thursday_r_pos_spy_all_std)
    thursday_comb_spy_all.append(thursday_r_pos_spy_all_cnt)
    thursday_comb_spy_all.append(thursday_r_comb_spy_all_mean)
    thursday_comb_spy_all.append(thursday_r_comb_spy_all_std)    

    ## Summarizing the result for SPY Ticker - friday
    friday_comb_spy_all=['Friday']
    
    friday_r_neg_spy_all_mean = round(statistics.mean(friday_all_neg_return_spy_all),5)
    friday_r_neg_spy_all_std = round(statistics.stdev(friday_all_neg_return_spy_all),5)
    friday_r_neg_spy_all_cnt = len(friday_all_neg_return_spy_all)
    friday_r_pos_spy_all_mean = round(statistics.mean(friday_all_pos_return_spy_all),5)
    friday_r_pos_spy_all_std = round(statistics.stdev(friday_all_pos_return_spy_all),5)
    friday_r_pos_spy_all_cnt = len(friday_all_pos_return_spy_all)
    friday_r_comb_spy_all_mean = round(statistics.mean(friday_comb_return_spy_all),5)
    friday_r_comb_spy_all_std = round(statistics.stdev(friday_comb_return_spy_all),5)
    
    # Appendig the result to empty string for friday SPY Ticker 
    friday_comb_spy_all.append(friday_r_neg_spy_all_mean)
    friday_comb_spy_all.append(friday_r_neg_spy_all_std)
    friday_comb_spy_all.append(friday_r_neg_spy_all_cnt)
    friday_comb_spy_all.append(friday_r_pos_spy_all_mean)
    friday_comb_spy_all.append(friday_r_pos_spy_all_std)
    friday_comb_spy_all.append(friday_r_pos_spy_all_cnt)
    friday_comb_spy_all.append(friday_r_comb_spy_all_mean)
    friday_comb_spy_all.append(friday_r_comb_spy_all_std)
    
    ## Printing the table for SPY Ticker 2014 to 2019
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_spy_all)
    t.add_row(tuesday_comb_spy_all)
    t.add_row(wednesday_comb_spy_all)
    t.add_row(thursday_comb_spy_all)
    t.add_row(friday_comb_spy_all)
    
    print('\n')
    print('Year- 2014 to 2019 SPY Ticker  \n')
    print(t)

    print('\n')    
    print("Question 3 Part 1. what is the best and worst days of the week for each?")
    print("Answer : According to me, best day is Tuesday and worst day is Monday for TM Ticker ")
    print('\n')    
    print("Answer : According to me, best day is Friday and worst day is Monday for SPY Ticker ")
    
    print('\n')    
    print("Question 3 Part 2.are these days the same for your stock as they are for S&P-500?")
    print("Answer : The daysare not same for my stock and the TM ticket, these are different ")
    
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)