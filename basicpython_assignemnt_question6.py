# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 6
Description of Problem (just a 1-2 line summary!): wrong advice from time to time.

######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 per given in pdf 

Step 1 : CSV file are read and casted to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , seperate master list is created , all further logic is derived off this master list 
Step 2a: The list is sorted using the sort function, this sorted list is used to derived the logic for part a b and c
Step 3a:  For part a -The 10 best are missed by starting the calculation from the 11tihe item
Step 3b:  For part b -The 10 worst are missed by using the original list and ignoring the negative returns
Step 3c:  For part c -The 5 best and 5 worst are missed by starting the calculation from the 5tihe item and for the 5 worst negative results are ignored
Step 4:  The calculation uses Amount*(1+Return), take only positive retrun while skip the negative returns
Step 5 : The logic goes through a for loop and exmaines the return for next to day to decide on the calcuation 
Step 6 : The aggregated simple computing result is stored in bucket variable which holds the final amount

    
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
    """    Code for Question 6 
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
    ##This logic was take up during the class
    ##
    ## $100 to start with to buy
    
    bucket=100
    
    # So the list is sorted with from highest to lowest return
    # Once the list is sorted , to miss the 10 best trading day the calculation starts with 11th best return till the end 
    # Per the Faciltator lecture on Friday, this is similar to queston 4 but wih missig the best 10 days
    data_line_all.sort(reverse=True)
    
    # Loop till before the last day since the preductin for last day if of the following day
    
    # Missing the best 10 best trading days 
    for i,line in  enumerate(data_line_all[10:]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part a : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker," with missed best 10 days is $",round(bucket,3)) 
    
    ## For SPY Ticker
    
    # creating empty list to segregate by years for TM Ticker from 2015 to 2019
    # makeing a list for 5 yrs from 2015 to 2019 
    
    data_line_all_spy=[]
        
    
    for line in  data_line_spy[1:]:
        #print(line[1])
        if float(line[1]) == 2015 or float(line[1]) == 2016 or float(line[1]) == 2017 or float(line[1]) == 2018 or float(line[1]) == 2019:       
            data_line_all_spy.append(line)
        else:
            pass
    
    
    
    ## The following is the assumption
    ## Oracle will tell us if the retrun is postive then hold on to it and if it is negative then skip those days
    ## The approach taken is to take Amount*(1+Return), take only positive retrun while skip the negative returns
    ##This logic was take up during the class
    ##
    ## $100 to start with to buy
    
    
    bucket=100
    
    # So the list is sorted with from highest to lowest return
    # Once the list is sorted , to miss the 10 best trading day the calculation starts with 11th best return till the end 
    # Per the Faciltator lecture on Friday, this is similar to queston 4 but wih missig the best 10 days
    data_line_all_spy.sort(reverse=True)
    
    # Loop till before the last day since the preductin for last day if of the following day
    
    # Missing the best 10 best trading days 
    for i,line in  enumerate(data_line_all_spy[10:]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all_spy[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all_spy[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part a : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker2," with missed best 10 days is $",round(bucket,3)) 
    
    ################ Part b 
    #### for TM ticker
    data_line_all.sort(reverse=True)
    bucket=100
    # Reusing the list from part a
    # Since Oracle gave you wrong results for worst 10 trading days, per prof we do not have to reverse the trasaction which mean we will are not going to trade on those days
    # The below code will ignore not do any trade on neg retruns 
    for i,line in  enumerate(data_line_all[:-1]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part b : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker," with missed best 10 days is $",round(bucket,3)) 
    
    #### for SPY ticker
    data_line_all_spy.sort(reverse=True)
    bucket=100
    # Reusing the list from part a
    # Since Oracle gave you wrong results for worst 10 trading days, per prof we do not have to reverse the trasaction which mean we will are not going to trade on those days
    # The below code will ignore not do any trade on neg retruns 
    for i,line in  enumerate(data_line_all_spy[:-1]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all_spy[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all_spy[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part b : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker2," with missed best 10 days is $",round(bucket,3)) 
   
    
   ################ Part c

    #### for TM ticker
    data_line_all.sort(reverse=True)
    bucket=100
    # Reusing the list from part a
    # Since Oracle gave you wrong results for worst 5 trading days, per prof we do not have to reverse the trasaction which mean we will are not going to trade on those days
    # The below code will ignore not do any trade on neg retruns 
    # The list starting point has changed from b
    for i,line in  enumerate(data_line_all[5:]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part c : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker," with missed best 10 days is $",round(bucket,3))     

    #### for SPY ticker
    data_line_all_spy.sort(reverse=True)
    bucket=100
    # Reusing the list from part a
    # Since Oracle gave you wrong results for worst 5 trading days, per prof we do not have to reverse the trasaction which mean we will are not going to trade on those days
    # The below code will ignore not do any trade on neg retruns 
    # The list starting point has changed from b
    for i,line in  enumerate(data_line_all_spy[5:]):
        
        # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
        # check for prediction of return for next day to buy or sell
        # if the return is negative skip it , use positive return only 
        
        if float(data_line_all_spy[i][-3]) > 0 :
            
            bucket=bucket*(1+float(data_line_all_spy[i][-3]))
            
    
        else:
            # Per Prof lecture , Oracle is giving prediction that on negative return you should not see/trade
            pass
    
    print('\n')       
    print("Question 6 part c : money you have with $100 start on 2015 to last trading day of 2019 for ticker",ticker2," with missed best 10 days is $",round(bucket,3)) 
    
    print('\n')   
    print(" Additional sub parts")
          
    print('\n')   
    print(" 1 : for each of the scenarios above (a,b and c), compute the_nal amount that you will have for both your stock and SPY")
    print("Answer - The amounts are shown above in various parts ")
    
    print('\n')   
    print(" 2 : do you gain more by missing the worst days or by missing the best days?")
    print("Answer - From the figure above, we gain more by missig the worst day, as we see from the numbers above if we miss the best days the amount is decreased ")
    
    print('\n')   
    print(" 3 : are the results in part (c) dierent from results that you obtained in question 4.")
    print("Answer - Yes, the numbers are different the gain are slightly lower in part c compared with question 4, the gain are higher in question 4 ")
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)    