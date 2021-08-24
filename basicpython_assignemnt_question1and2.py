# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 1 and Question 2
Description of Problem (just a 1-2 line summary!): Create summary table for each ticker
    
######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2014 to 2019 per given in pdf 

Step 1 : CSV file are read and casted to a list , one for TM ticker and other for SPY ticker
Step 3 : For each ticker , seperate master list is created , all further logic is derived off this master list 
Step 4 : For each year from 2014 to 2019 it broken down in to seperate list 
Step 5 : For each year , for each day Monday...to Friday , again seperate sublist is created further
Step 6 : Within each sublist further sublist is created for every attribute asked in the question 
Step 7 : For calcuation statistics package is used for calculation of mean and standard deviation
Step 8 : For printing output table "PrettyTable" is used for easy readibility 
        
     >>> Please expand the right Console the see the output properly  

Please note summary for year 2014 is also done and printed in the output although the question is only asking for 2015 to 2019    
"""

import os
import csv
import statistics

# run this  ! pip install PrettyTable
from prettytable import PrettyTable

ticker='TM'
here =os.path.abspath( __file__ )
input_dir =os.path.abspath(os.path.join(here ,os. pardir ))
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    """    Code for Question 1 & 2
    """
        
    
    # using uft encoding 
    # Casting the line of xls to a list 
    data= open(ticker_file,encoding='utf-8')
    csv_data = csv.reader(data, delimiter=',')
    data_line= list(csv_data)
    
    
    # creating empty list to segregate by years 
     
    data_line_2014=[]
    data_line_2015=[]
    data_line_2016=[]
    data_line_2017=[]
    data_line_2018=[]
    data_line_2019=[]
    
    for line in  data_line[1:]:
        
        if float(line[1]) == 2014:
            data_line_2014.append(line)            
        elif float(line[1]) == 2015:
            data_line_2015.append(line)
        elif float(line[1]) == 2016:
            data_line_2016.append(line)    
        elif float(line[1]) == 2017:
             data_line_2017.append(line)   
        elif float(line[1]) == 2018:
            data_line_2018.append(line)  
        else:
             data_line_2019.append(line)   
           
        
    # For year 2014 Creating list for Daily return, Negative returns and pos return 
    
    monday_all_neg_return_2014=[]
    monday_all_pos_return_2014=[]
    monday_comb_return_2014=[]
    tuesday_all_neg_return_2014=[]
    tuesday_all_pos_return_2014=[]
    tuesday_comb_return_2014=[]
    wednesday_all_neg_return_2014=[]
    wednesday_all_pos_return_2014=[]
    wednesday_comb_return_2014=[]
    thursday_all_neg_return_2014=[]
    thursday_all_pos_return_2014=[]
    thursday_comb_return_2014=[]
    friday_all_neg_return_2014=[]
    friday_all_pos_return_2014=[]
    friday_comb_return_2014=[]

    for line in  data_line_2014[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2014.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2014.append(float(line[-3]))
            else:
                monday_all_neg_return_2014.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2014.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2014.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2014.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2014.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2014.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2014.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2014.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2014.append(float(line[-3]))
            else:
                thursday_all_neg_return_2014.append(float(line[-3]))        
        else :
            friday_comb_return_2014.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2014.append(float(line[-3]))
            else:
                friday_all_neg_return_2014.append(float(line[-3]))        
        
    ## Summarizing the result for 2014- Monday
    monday_comb_2014=['Monday']
    
    monday_r_neg_2014_mean = round(statistics.mean(monday_all_neg_return_2014),5)
    monday_r_neg_2014_std = round(statistics.stdev(monday_all_neg_return_2014),5)
    monday_r_neg_2014_cnt = len(monday_all_neg_return_2014)
    monday_r_pos_2014_mean = round(statistics.mean(monday_all_pos_return_2014),5)
    monday_r_pos_2014_std = round(statistics.stdev(monday_all_pos_return_2014),5)
    monday_r_pos_2014_cnt = len(monday_all_pos_return_2014)
    monday_r_comb_2014_mean = round(statistics.mean(monday_comb_return_2014),5)
    monday_r_comb_2014_std = round(statistics.stdev(monday_comb_return_2014),5)
    
    # Appendig the result to empty string for Monday 2014
    monday_comb_2014.append(monday_r_neg_2014_mean)
    monday_comb_2014.append(monday_r_neg_2014_std)
    monday_comb_2014.append(monday_r_neg_2014_cnt)
    monday_comb_2014.append(monday_r_pos_2014_mean)
    monday_comb_2014.append(monday_r_pos_2014_std)
    monday_comb_2014.append(monday_r_pos_2014_cnt)
    monday_comb_2014.append(monday_r_comb_2014_mean)
    monday_comb_2014.append(monday_r_comb_2014_std)

    ## Summarizing the result for 2014- tuesday
    tuesday_comb_2014=['Tuesday']
    
    tuesday_r_neg_2014_mean = round(statistics.mean(tuesday_all_neg_return_2014),5)
    tuesday_r_neg_2014_std = round(statistics.stdev(tuesday_all_neg_return_2014),5)
    tuesday_r_neg_2014_cnt = len(tuesday_all_neg_return_2014)
    tuesday_r_pos_2014_mean = round(statistics.mean(tuesday_all_pos_return_2014),5)
    tuesday_r_pos_2014_std = round(statistics.stdev(tuesday_all_pos_return_2014),5)
    tuesday_r_pos_2014_cnt = len(tuesday_all_pos_return_2014)
    tuesday_r_comb_2014_mean = round(statistics.mean(tuesday_comb_return_2014),5)
    tuesday_r_comb_2014_std = round(statistics.stdev(tuesday_comb_return_2014),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2014.append(tuesday_r_neg_2014_mean)
    tuesday_comb_2014.append(tuesday_r_neg_2014_std)
    tuesday_comb_2014.append(tuesday_r_neg_2014_cnt)
    tuesday_comb_2014.append(tuesday_r_pos_2014_mean)
    tuesday_comb_2014.append(tuesday_r_pos_2014_std)
    tuesday_comb_2014.append(tuesday_r_pos_2014_cnt)
    tuesday_comb_2014.append(tuesday_r_comb_2014_mean)
    tuesday_comb_2014.append(tuesday_r_comb_2014_std)    
    
    ## Summarizing the result for 2014- wednesday
    wednesday_comb_2014=['Wednesday']
    
    wednesday_r_neg_2014_mean = round(statistics.mean(wednesday_all_neg_return_2014),5)
    wednesday_r_neg_2014_std = round(statistics.stdev(wednesday_all_neg_return_2014),5)
    wednesday_r_neg_2014_cnt = len(wednesday_all_neg_return_2014)
    wednesday_r_pos_2014_mean = round(statistics.mean(wednesday_all_pos_return_2014),5)
    wednesday_r_pos_2014_std = round(statistics.stdev(wednesday_all_pos_return_2014),5)
    wednesday_r_pos_2014_cnt = len(wednesday_all_pos_return_2014)
    wednesday_r_comb_2014_mean = round(statistics.mean(wednesday_comb_return_2014),5)
    wednesday_r_comb_2014_std = round(statistics.stdev(wednesday_comb_return_2014),5)
    
    # Appendig the result to empty string for wednesday 2014
    wednesday_comb_2014.append(wednesday_r_neg_2014_mean)
    wednesday_comb_2014.append(wednesday_r_neg_2014_std)
    wednesday_comb_2014.append(wednesday_r_neg_2014_cnt)
    wednesday_comb_2014.append(wednesday_r_pos_2014_mean)
    wednesday_comb_2014.append(wednesday_r_pos_2014_std)
    wednesday_comb_2014.append(wednesday_r_pos_2014_cnt)
    wednesday_comb_2014.append(wednesday_r_comb_2014_mean)
    wednesday_comb_2014.append(wednesday_r_comb_2014_std)
    
    ## Summarizing the result for 2014- thursday
    thursday_comb_2014=['Thursday']
    
    thursday_r_neg_2014_mean = round(statistics.mean(thursday_all_neg_return_2014),5)
    thursday_r_neg_2014_std = round(statistics.stdev(thursday_all_neg_return_2014),5)
    thursday_r_neg_2014_cnt = len(thursday_all_neg_return_2014)
    thursday_r_pos_2014_mean = round(statistics.mean(thursday_all_pos_return_2014),5)
    thursday_r_pos_2014_std = round(statistics.stdev(thursday_all_pos_return_2014),5)
    thursday_r_pos_2014_cnt = len(thursday_all_pos_return_2014)
    thursday_r_comb_2014_mean = round(statistics.mean(thursday_comb_return_2014),5)
    thursday_r_comb_2014_std = round(statistics.stdev(thursday_comb_return_2014),5)
    
    # Appendig the result to empty string for thursday 2014
    thursday_comb_2014.append(thursday_r_neg_2014_mean)
    thursday_comb_2014.append(thursday_r_neg_2014_std)
    thursday_comb_2014.append(thursday_r_neg_2014_cnt)
    thursday_comb_2014.append(thursday_r_pos_2014_mean)
    thursday_comb_2014.append(thursday_r_pos_2014_std)
    thursday_comb_2014.append(thursday_r_pos_2014_cnt)
    thursday_comb_2014.append(thursday_r_comb_2014_mean)
    thursday_comb_2014.append(thursday_r_comb_2014_std)    

    ## Summarizing the result for 2014- friday
    friday_comb_2014=['Friday']
    
    friday_r_neg_2014_mean = round(statistics.mean(friday_all_neg_return_2014),5)
    friday_r_neg_2014_std = round(statistics.stdev(friday_all_neg_return_2014),5)
    friday_r_neg_2014_cnt = len(friday_all_neg_return_2014)
    friday_r_pos_2014_mean = round(statistics.mean(friday_all_pos_return_2014),5)
    friday_r_pos_2014_std = round(statistics.stdev(friday_all_pos_return_2014),5)
    friday_r_pos_2014_cnt = len(friday_all_pos_return_2014)
    friday_r_comb_2014_mean = round(statistics.mean(friday_comb_return_2014),5)
    friday_r_comb_2014_std = round(statistics.stdev(friday_comb_return_2014),5)
    
    # Appendig the result to empty string for friday 2014
    friday_comb_2014.append(friday_r_neg_2014_mean)
    friday_comb_2014.append(friday_r_neg_2014_std)
    friday_comb_2014.append(friday_r_neg_2014_cnt)
    friday_comb_2014.append(friday_r_pos_2014_mean)
    friday_comb_2014.append(friday_r_pos_2014_std)
    friday_comb_2014.append(friday_r_pos_2014_cnt)
    friday_comb_2014.append(friday_r_comb_2014_mean)
    friday_comb_2014.append(friday_r_comb_2014_std)
    
    ## Printing the table for 2014
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2014)
    t.add_row(tuesday_comb_2014)
    t.add_row(wednesday_comb_2014)
    t.add_row(thursday_comb_2014)
    t.add_row(friday_comb_2014)
    
    print('\n')
    print('Year 2014 \n')
    print(t)
    
        
   # for 2015 Creating list for Daily return, Negative returns and pos return


    monday_all_neg_return_2015=[]
    monday_all_pos_return_2015=[]
    monday_comb_return_2015=[]
    tuesday_all_neg_return_2015=[]
    tuesday_all_pos_return_2015=[]
    tuesday_comb_return_2015=[]
    wednesday_all_neg_return_2015=[]
    wednesday_all_pos_return_2015=[]
    wednesday_comb_return_2015=[]
    thursday_all_neg_return_2015=[]
    thursday_all_pos_return_2015=[]
    thursday_comb_return_2015=[]
    friday_all_neg_return_2015=[]
    friday_all_pos_return_2015=[]
    friday_comb_return_2015=[]

    for line in  data_line_2015[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2015.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2015.append(float(line[-3]))
            else:
                monday_all_neg_return_2015.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2015.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2015.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2015.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2015.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2015.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2015.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2015.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2015.append(float(line[-3]))
            else:
                thursday_all_neg_return_2015.append(float(line[-3]))        
        else :
            friday_comb_return_2015.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2015.append(float(line[-3]))
            else:
                friday_all_neg_return_2015.append(float(line[-3]))        
        
    ## Summarizing the result for 2015- Monday
    monday_comb_2015=['Monday']
    
    monday_r_neg_2015_mean = round(statistics.mean(monday_all_neg_return_2015),5)
    monday_r_neg_2015_std = round(statistics.stdev(monday_all_neg_return_2015),5)
    monday_r_neg_2015_cnt = len(monday_all_neg_return_2015)
    monday_r_pos_2015_mean = round(statistics.mean(monday_all_pos_return_2015),5)
    monday_r_pos_2015_std = round(statistics.stdev(monday_all_pos_return_2015),5)
    monday_r_pos_2015_cnt = len(monday_all_pos_return_2015)
    monday_r_comb_2015_mean = round(statistics.mean(monday_comb_return_2015),5)
    monday_r_comb_2015_std = round(statistics.stdev(monday_comb_return_2015),5)
    
    # Appendig the result to empty string for Monday 2015
    monday_comb_2015.append(monday_r_neg_2015_mean)
    monday_comb_2015.append(monday_r_neg_2015_std)
    monday_comb_2015.append(monday_r_neg_2015_cnt)
    monday_comb_2015.append(monday_r_pos_2015_mean)
    monday_comb_2015.append(monday_r_pos_2015_std)
    monday_comb_2015.append(monday_r_pos_2015_cnt)
    monday_comb_2015.append(monday_r_comb_2015_mean)
    monday_comb_2015.append(monday_r_comb_2015_std)

    ## Summarizing the result for 2015- tuesday
    tuesday_comb_2015=['Tuesday']
    
    tuesday_r_neg_2015_mean = round(statistics.mean(tuesday_all_neg_return_2015),5)
    tuesday_r_neg_2015_std = round(statistics.stdev(tuesday_all_neg_return_2015),5)
    tuesday_r_neg_2015_cnt = len(tuesday_all_neg_return_2015)
    tuesday_r_pos_2015_mean = round(statistics.mean(tuesday_all_pos_return_2015),5)
    tuesday_r_pos_2015_std = round(statistics.stdev(tuesday_all_pos_return_2015),5)
    tuesday_r_pos_2015_cnt = len(tuesday_all_pos_return_2015)
    tuesday_r_comb_2015_mean = round(statistics.mean(tuesday_comb_return_2015),5)
    tuesday_r_comb_2015_std = round(statistics.stdev(tuesday_comb_return_2015),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2015.append(tuesday_r_neg_2015_mean)
    tuesday_comb_2015.append(tuesday_r_neg_2015_std)
    tuesday_comb_2015.append(tuesday_r_neg_2015_cnt)
    tuesday_comb_2015.append(tuesday_r_pos_2015_mean)
    tuesday_comb_2015.append(tuesday_r_pos_2015_std)
    tuesday_comb_2015.append(tuesday_r_pos_2015_cnt)
    tuesday_comb_2015.append(tuesday_r_comb_2015_mean)
    tuesday_comb_2015.append(tuesday_r_comb_2015_std)    
    
    ## Summarizing the result for 2015- wednesday
    wednesday_comb_2015=['Wednesday']
    
    wednesday_r_neg_2015_mean = round(statistics.mean(wednesday_all_neg_return_2015),5)
    wednesday_r_neg_2015_std = round(statistics.stdev(wednesday_all_neg_return_2015),5)
    wednesday_r_neg_2015_cnt = len(wednesday_all_neg_return_2015)
    wednesday_r_pos_2015_mean = round(statistics.mean(wednesday_all_pos_return_2015),5)
    wednesday_r_pos_2015_std = round(statistics.stdev(wednesday_all_pos_return_2015),5)
    wednesday_r_pos_2015_cnt = len(wednesday_all_pos_return_2015)
    wednesday_r_comb_2015_mean = round(statistics.mean(wednesday_comb_return_2015),5)
    wednesday_r_comb_2015_std = round(statistics.stdev(wednesday_comb_return_2015),5)
    
    # Appendig the result to empty string for wednesday 2015
    wednesday_comb_2015.append(wednesday_r_neg_2015_mean)
    wednesday_comb_2015.append(wednesday_r_neg_2015_std)
    wednesday_comb_2015.append(wednesday_r_neg_2015_cnt)
    wednesday_comb_2015.append(wednesday_r_pos_2015_mean)
    wednesday_comb_2015.append(wednesday_r_pos_2015_std)
    wednesday_comb_2015.append(wednesday_r_pos_2015_cnt)
    wednesday_comb_2015.append(wednesday_r_comb_2015_mean)
    wednesday_comb_2015.append(wednesday_r_comb_2015_std)
    
    ## Summarizing the result for 2015- thursday
    thursday_comb_2015=['Thursday']
    
    thursday_r_neg_2015_mean = round(statistics.mean(thursday_all_neg_return_2015),5)
    thursday_r_neg_2015_std = round(statistics.stdev(thursday_all_neg_return_2015),5)
    thursday_r_neg_2015_cnt = len(thursday_all_neg_return_2015)
    thursday_r_pos_2015_mean = round(statistics.mean(thursday_all_pos_return_2015),5)
    thursday_r_pos_2015_std = round(statistics.stdev(thursday_all_pos_return_2015),5)
    thursday_r_pos_2015_cnt = len(thursday_all_pos_return_2015)
    thursday_r_comb_2015_mean = round(statistics.mean(thursday_comb_return_2015),5)
    thursday_r_comb_2015_std = round(statistics.stdev(thursday_comb_return_2015),5)
    
    # Appendig the result to empty string for thursday 2015
    thursday_comb_2015.append(thursday_r_neg_2015_mean)
    thursday_comb_2015.append(thursday_r_neg_2015_std)
    thursday_comb_2015.append(thursday_r_neg_2015_cnt)
    thursday_comb_2015.append(thursday_r_pos_2015_mean)
    thursday_comb_2015.append(thursday_r_pos_2015_std)
    thursday_comb_2015.append(thursday_r_pos_2015_cnt)
    thursday_comb_2015.append(thursday_r_comb_2015_mean)
    thursday_comb_2015.append(thursday_r_comb_2015_std)    

    ## Summarizing the result for 2015- friday
    friday_comb_2015=['Friday']
    
    friday_r_neg_2015_mean = round(statistics.mean(friday_all_neg_return_2015),5)
    friday_r_neg_2015_std = round(statistics.stdev(friday_all_neg_return_2015),5)
    friday_r_neg_2015_cnt = len(friday_all_neg_return_2015)
    friday_r_pos_2015_mean = round(statistics.mean(friday_all_pos_return_2015),5)
    friday_r_pos_2015_std = round(statistics.stdev(friday_all_pos_return_2015),5)
    friday_r_pos_2015_cnt = len(friday_all_pos_return_2015)
    friday_r_comb_2015_mean = round(statistics.mean(friday_comb_return_2015),5)
    friday_r_comb_2015_std = round(statistics.stdev(friday_comb_return_2015),5)
    
    # Appendig the result to empty string for friday 2015
    friday_comb_2015.append(friday_r_neg_2015_mean)
    friday_comb_2015.append(friday_r_neg_2015_std)
    friday_comb_2015.append(friday_r_neg_2015_cnt)
    friday_comb_2015.append(friday_r_pos_2015_mean)
    friday_comb_2015.append(friday_r_pos_2015_std)
    friday_comb_2015.append(friday_r_pos_2015_cnt)
    friday_comb_2015.append(friday_r_comb_2015_mean)
    friday_comb_2015.append(friday_r_comb_2015_std)
    
    ## Printing the table for 2015
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2015)
    t.add_row(tuesday_comb_2015)
    t.add_row(wednesday_comb_2015)
    t.add_row(thursday_comb_2015)
    t.add_row(friday_comb_2015)
    
    print('\n')
    print('Year 2015 \n')
    print(t)

    
   # Creating list for Daily return, Negative returns and pos return for 2016

    monday_all_neg_return_2016=[]
    monday_all_pos_return_2016=[]
    monday_comb_return_2016=[]
    tuesday_all_neg_return_2016=[]
    tuesday_all_pos_return_2016=[]
    tuesday_comb_return_2016=[]
    wednesday_all_neg_return_2016=[]
    wednesday_all_pos_return_2016=[]
    wednesday_comb_return_2016=[]
    thursday_all_neg_return_2016=[]
    thursday_all_pos_return_2016=[]
    thursday_comb_return_2016=[]
    friday_all_neg_return_2016=[]
    friday_all_pos_return_2016=[]
    friday_comb_return_2016=[]

    for line in  data_line_2016[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2016.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2016.append(float(line[-3]))
            else:
                monday_all_neg_return_2016.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2016.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2016.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2016.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2016.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2016.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2016.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2016.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2016.append(float(line[-3]))
            else:
                thursday_all_neg_return_2016.append(float(line[-3]))        
        else :
            friday_comb_return_2016.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2016.append(float(line[-3]))
            else:
                friday_all_neg_return_2016.append(float(line[-3]))        
        
    ## Summarizing the result for 2016- Monday
    monday_comb_2016=['Monday']
    
    monday_r_neg_2016_mean = round(statistics.mean(monday_all_neg_return_2016),5)
    monday_r_neg_2016_std = round(statistics.stdev(monday_all_neg_return_2016),5)
    monday_r_neg_2016_cnt = len(monday_all_neg_return_2016)
    monday_r_pos_2016_mean = round(statistics.mean(monday_all_pos_return_2016),5)
    monday_r_pos_2016_std = round(statistics.stdev(monday_all_pos_return_2016),5)
    monday_r_pos_2016_cnt = len(monday_all_pos_return_2016)
    monday_r_comb_2016_mean = round(statistics.mean(monday_comb_return_2016),5)
    monday_r_comb_2016_std = round(statistics.stdev(monday_comb_return_2016),5)
    
    # Appendig the result to empty string for Monday 2016
    monday_comb_2016.append(monday_r_neg_2016_mean)
    monday_comb_2016.append(monday_r_neg_2016_std)
    monday_comb_2016.append(monday_r_neg_2016_cnt)
    monday_comb_2016.append(monday_r_pos_2016_mean)
    monday_comb_2016.append(monday_r_pos_2016_std)
    monday_comb_2016.append(monday_r_pos_2016_cnt)
    monday_comb_2016.append(monday_r_comb_2016_mean)
    monday_comb_2016.append(monday_r_comb_2016_std)

    ## Summarizing the result for 2016- tuesday
    tuesday_comb_2016=['Tuesday']
    
    tuesday_r_neg_2016_mean = round(statistics.mean(tuesday_all_neg_return_2016),5)
    tuesday_r_neg_2016_std = round(statistics.stdev(tuesday_all_neg_return_2016),5)
    tuesday_r_neg_2016_cnt = len(tuesday_all_neg_return_2016)
    tuesday_r_pos_2016_mean = round(statistics.mean(tuesday_all_pos_return_2016),5)
    tuesday_r_pos_2016_std = round(statistics.stdev(tuesday_all_pos_return_2016),5)
    tuesday_r_pos_2016_cnt = len(tuesday_all_pos_return_2016)
    tuesday_r_comb_2016_mean = round(statistics.mean(tuesday_comb_return_2016),5)
    tuesday_r_comb_2016_std = round(statistics.stdev(tuesday_comb_return_2016),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2016.append(tuesday_r_neg_2016_mean)
    tuesday_comb_2016.append(tuesday_r_neg_2016_std)
    tuesday_comb_2016.append(tuesday_r_neg_2016_cnt)
    tuesday_comb_2016.append(tuesday_r_pos_2016_mean)
    tuesday_comb_2016.append(tuesday_r_pos_2016_std)
    tuesday_comb_2016.append(tuesday_r_pos_2016_cnt)
    tuesday_comb_2016.append(tuesday_r_comb_2016_mean)
    tuesday_comb_2016.append(tuesday_r_comb_2016_std)    
    
    ## Summarizing the result for 2016- wednesday
    wednesday_comb_2016=['Wednesday']
    
    wednesday_r_neg_2016_mean = round(statistics.mean(wednesday_all_neg_return_2016),5)
    wednesday_r_neg_2016_std = round(statistics.stdev(wednesday_all_neg_return_2016),5)
    wednesday_r_neg_2016_cnt = len(wednesday_all_neg_return_2016)
    wednesday_r_pos_2016_mean = round(statistics.mean(wednesday_all_pos_return_2016),5)
    wednesday_r_pos_2016_std = round(statistics.stdev(wednesday_all_pos_return_2016),5)
    wednesday_r_pos_2016_cnt = len(wednesday_all_pos_return_2016)
    wednesday_r_comb_2016_mean = round(statistics.mean(wednesday_comb_return_2016),5)
    wednesday_r_comb_2016_std = round(statistics.stdev(wednesday_comb_return_2016),5)
    
    # Appendig the result to empty string for wednesday 2016
    wednesday_comb_2016.append(wednesday_r_neg_2016_mean)
    wednesday_comb_2016.append(wednesday_r_neg_2016_std)
    wednesday_comb_2016.append(wednesday_r_neg_2016_cnt)
    wednesday_comb_2016.append(wednesday_r_pos_2016_mean)
    wednesday_comb_2016.append(wednesday_r_pos_2016_std)
    wednesday_comb_2016.append(wednesday_r_pos_2016_cnt)
    wednesday_comb_2016.append(wednesday_r_comb_2016_mean)
    wednesday_comb_2016.append(wednesday_r_comb_2016_std)
    
    ## Summarizing the result for 2016- thursday
    thursday_comb_2016=['Thursday']
    
    thursday_r_neg_2016_mean = round(statistics.mean(thursday_all_neg_return_2016),5)
    thursday_r_neg_2016_std = round(statistics.stdev(thursday_all_neg_return_2016),5)
    thursday_r_neg_2016_cnt = len(thursday_all_neg_return_2016)
    thursday_r_pos_2016_mean = round(statistics.mean(thursday_all_pos_return_2016),5)
    thursday_r_pos_2016_std = round(statistics.stdev(thursday_all_pos_return_2016),5)
    thursday_r_pos_2016_cnt = len(thursday_all_pos_return_2016)
    thursday_r_comb_2016_mean = round(statistics.mean(thursday_comb_return_2016),5)
    thursday_r_comb_2016_std = round(statistics.stdev(thursday_comb_return_2016),5)
    
    # Appendig the result to empty string for thursday 2016
    thursday_comb_2016.append(thursday_r_neg_2016_mean)
    thursday_comb_2016.append(thursday_r_neg_2016_std)
    thursday_comb_2016.append(thursday_r_neg_2016_cnt)
    thursday_comb_2016.append(thursday_r_pos_2016_mean)
    thursday_comb_2016.append(thursday_r_pos_2016_std)
    thursday_comb_2016.append(thursday_r_pos_2016_cnt)
    thursday_comb_2016.append(thursday_r_comb_2016_mean)
    thursday_comb_2016.append(thursday_r_comb_2016_std)    

    ## Summarizing the result for 2016- friday
    friday_comb_2016=['Friday']
    
    friday_r_neg_2016_mean = round(statistics.mean(friday_all_neg_return_2016),5)
    friday_r_neg_2016_std = round(statistics.stdev(friday_all_neg_return_2016),5)
    friday_r_neg_2016_cnt = len(friday_all_neg_return_2016)
    friday_r_pos_2016_mean = round(statistics.mean(friday_all_pos_return_2016),5)
    friday_r_pos_2016_std = round(statistics.stdev(friday_all_pos_return_2016),5)
    friday_r_pos_2016_cnt = len(friday_all_pos_return_2016)
    friday_r_comb_2016_mean = round(statistics.mean(friday_comb_return_2016),5)
    friday_r_comb_2016_std = round(statistics.stdev(friday_comb_return_2016),5)
    
    # Appendig the result to empty string for friday 2016
    friday_comb_2016.append(friday_r_neg_2016_mean)
    friday_comb_2016.append(friday_r_neg_2016_std)
    friday_comb_2016.append(friday_r_neg_2016_cnt)
    friday_comb_2016.append(friday_r_pos_2016_mean)
    friday_comb_2016.append(friday_r_pos_2016_std)
    friday_comb_2016.append(friday_r_pos_2016_cnt)
    friday_comb_2016.append(friday_r_comb_2016_mean)
    friday_comb_2016.append(friday_r_comb_2016_std)
    
    ## Printing the table for 2016
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2016)
    t.add_row(tuesday_comb_2016)
    t.add_row(wednesday_comb_2016)
    t.add_row(thursday_comb_2016)
    t.add_row(friday_comb_2016)
    
    print('\n')
    print('Year 2016 \n')
    print(t)
            
       
   # Creating list for Daily return, Negative returns and pos return for 2017

    monday_all_neg_return_2017=[]
    monday_all_pos_return_2017=[]
    monday_comb_return_2017=[]
    tuesday_all_neg_return_2017=[]
    tuesday_all_pos_return_2017=[]
    tuesday_comb_return_2017=[]
    wednesday_all_neg_return_2017=[]
    wednesday_all_pos_return_2017=[]
    wednesday_comb_return_2017=[]
    thursday_all_neg_return_2017=[]
    thursday_all_pos_return_2017=[]
    thursday_comb_return_2017=[]
    friday_all_neg_return_2017=[]
    friday_all_pos_return_2017=[]
    friday_comb_return_2017=[]

    for line in  data_line_2017[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2017.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2017.append(float(line[-3]))
            else:
                monday_all_neg_return_2017.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2017.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2017.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2017.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2017.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2017.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2017.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2017.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2017.append(float(line[-3]))
            else:
                thursday_all_neg_return_2017.append(float(line[-3]))        
        else :
            friday_comb_return_2017.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2017.append(float(line[-3]))
            else:
                friday_all_neg_return_2017.append(float(line[-3]))        
        
    ## Summarizing the result for 2017- Monday
    monday_comb_2017=['Monday']
    
    monday_r_neg_2017_mean = round(statistics.mean(monday_all_neg_return_2017),5)
    monday_r_neg_2017_std = round(statistics.stdev(monday_all_neg_return_2017),5)
    monday_r_neg_2017_cnt = len(monday_all_neg_return_2017)
    monday_r_pos_2017_mean = round(statistics.mean(monday_all_pos_return_2017),5)
    monday_r_pos_2017_std = round(statistics.stdev(monday_all_pos_return_2017),5)
    monday_r_pos_2017_cnt = len(monday_all_pos_return_2017)
    monday_r_comb_2017_mean = round(statistics.mean(monday_comb_return_2017),5)
    monday_r_comb_2017_std = round(statistics.stdev(monday_comb_return_2017),5)
    
    # Appendig the result to empty string for Monday 2017
    monday_comb_2017.append(monday_r_neg_2017_mean)
    monday_comb_2017.append(monday_r_neg_2017_std)
    monday_comb_2017.append(monday_r_neg_2017_cnt)
    monday_comb_2017.append(monday_r_pos_2017_mean)
    monday_comb_2017.append(monday_r_pos_2017_std)
    monday_comb_2017.append(monday_r_pos_2017_cnt)
    monday_comb_2017.append(monday_r_comb_2017_mean)
    monday_comb_2017.append(monday_r_comb_2017_std)

    ## Summarizing the result for 2017- tuesday
    tuesday_comb_2017=['Tuesday']
    
    tuesday_r_neg_2017_mean = round(statistics.mean(tuesday_all_neg_return_2017),5)
    tuesday_r_neg_2017_std = round(statistics.stdev(tuesday_all_neg_return_2017),5)
    tuesday_r_neg_2017_cnt = len(tuesday_all_neg_return_2017)
    tuesday_r_pos_2017_mean = round(statistics.mean(tuesday_all_pos_return_2017),5)
    tuesday_r_pos_2017_std = round(statistics.stdev(tuesday_all_pos_return_2017),5)
    tuesday_r_pos_2017_cnt = len(tuesday_all_pos_return_2017)
    tuesday_r_comb_2017_mean = round(statistics.mean(tuesday_comb_return_2017),5)
    tuesday_r_comb_2017_std = round(statistics.stdev(tuesday_comb_return_2017),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2017.append(tuesday_r_neg_2017_mean)
    tuesday_comb_2017.append(tuesday_r_neg_2017_std)
    tuesday_comb_2017.append(tuesday_r_neg_2017_cnt)
    tuesday_comb_2017.append(tuesday_r_pos_2017_mean)
    tuesday_comb_2017.append(tuesday_r_pos_2017_std)
    tuesday_comb_2017.append(tuesday_r_pos_2017_cnt)
    tuesday_comb_2017.append(tuesday_r_comb_2017_mean)
    tuesday_comb_2017.append(tuesday_r_comb_2017_std)    
    
    ## Summarizing the result for 2017- wednesday
    wednesday_comb_2017=['Wednesday']
    
    wednesday_r_neg_2017_mean = round(statistics.mean(wednesday_all_neg_return_2017),5)
    wednesday_r_neg_2017_std = round(statistics.stdev(wednesday_all_neg_return_2017),5)
    wednesday_r_neg_2017_cnt = len(wednesday_all_neg_return_2017)
    wednesday_r_pos_2017_mean = round(statistics.mean(wednesday_all_pos_return_2017),5)
    wednesday_r_pos_2017_std = round(statistics.stdev(wednesday_all_pos_return_2017),5)
    wednesday_r_pos_2017_cnt = len(wednesday_all_pos_return_2017)
    wednesday_r_comb_2017_mean = round(statistics.mean(wednesday_comb_return_2017),5)
    wednesday_r_comb_2017_std = round(statistics.stdev(wednesday_comb_return_2017),5)
    
    # Appendig the result to empty string for wednesday 2017
    wednesday_comb_2017.append(wednesday_r_neg_2017_mean)
    wednesday_comb_2017.append(wednesday_r_neg_2017_std)
    wednesday_comb_2017.append(wednesday_r_neg_2017_cnt)
    wednesday_comb_2017.append(wednesday_r_pos_2017_mean)
    wednesday_comb_2017.append(wednesday_r_pos_2017_std)
    wednesday_comb_2017.append(wednesday_r_pos_2017_cnt)
    wednesday_comb_2017.append(wednesday_r_comb_2017_mean)
    wednesday_comb_2017.append(wednesday_r_comb_2017_std)
    
    ## Summarizing the result for 2017- thursday
    thursday_comb_2017=['Thursday']
    
    thursday_r_neg_2017_mean = round(statistics.mean(thursday_all_neg_return_2017),5)
    thursday_r_neg_2017_std = round(statistics.stdev(thursday_all_neg_return_2017),5)
    thursday_r_neg_2017_cnt = len(thursday_all_neg_return_2017)
    thursday_r_pos_2017_mean = round(statistics.mean(thursday_all_pos_return_2017),5)
    thursday_r_pos_2017_std = round(statistics.stdev(thursday_all_pos_return_2017),5)
    thursday_r_pos_2017_cnt = len(thursday_all_pos_return_2017)
    thursday_r_comb_2017_mean = round(statistics.mean(thursday_comb_return_2017),5)
    thursday_r_comb_2017_std = round(statistics.stdev(thursday_comb_return_2017),5)
    
    # Appendig the result to empty string for thursday 2017
    thursday_comb_2017.append(thursday_r_neg_2017_mean)
    thursday_comb_2017.append(thursday_r_neg_2017_std)
    thursday_comb_2017.append(thursday_r_neg_2017_cnt)
    thursday_comb_2017.append(thursday_r_pos_2017_mean)
    thursday_comb_2017.append(thursday_r_pos_2017_std)
    thursday_comb_2017.append(thursday_r_pos_2017_cnt)
    thursday_comb_2017.append(thursday_r_comb_2017_mean)
    thursday_comb_2017.append(thursday_r_comb_2017_std)    

    ## Summarizing the result for 2017- friday
    friday_comb_2017=['Friday']
    
    friday_r_neg_2017_mean = round(statistics.mean(friday_all_neg_return_2017),5)
    friday_r_neg_2017_std = round(statistics.stdev(friday_all_neg_return_2017),5)
    friday_r_neg_2017_cnt = len(friday_all_neg_return_2017)
    friday_r_pos_2017_mean = round(statistics.mean(friday_all_pos_return_2017),5)
    friday_r_pos_2017_std = round(statistics.stdev(friday_all_pos_return_2017),5)
    friday_r_pos_2017_cnt = len(friday_all_pos_return_2017)
    friday_r_comb_2017_mean = round(statistics.mean(friday_comb_return_2017),5)
    friday_r_comb_2017_std = round(statistics.stdev(friday_comb_return_2017),5)
    
    # Appendig the result to empty string for friday 2017
    friday_comb_2017.append(friday_r_neg_2017_mean)
    friday_comb_2017.append(friday_r_neg_2017_std)
    friday_comb_2017.append(friday_r_neg_2017_cnt)
    friday_comb_2017.append(friday_r_pos_2017_mean)
    friday_comb_2017.append(friday_r_pos_2017_std)
    friday_comb_2017.append(friday_r_pos_2017_cnt)
    friday_comb_2017.append(friday_r_comb_2017_mean)
    friday_comb_2017.append(friday_r_comb_2017_std)
    
    ## Printing the table for 2017
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2017)
    t.add_row(tuesday_comb_2017)
    t.add_row(wednesday_comb_2017)
    t.add_row(thursday_comb_2017)
    t.add_row(friday_comb_2017)
    
    print('\n')
    print('Year 2017 \n')
    print(t)

   # Creating list for Daily return, Negative returns and pos return for 2018

    monday_all_neg_return_2018=[]
    monday_all_pos_return_2018=[]
    monday_comb_return_2018=[]
    tuesday_all_neg_return_2018=[]
    tuesday_all_pos_return_2018=[]
    tuesday_comb_return_2018=[]
    wednesday_all_neg_return_2018=[]
    wednesday_all_pos_return_2018=[]
    wednesday_comb_return_2018=[]
    thursday_all_neg_return_2018=[]
    thursday_all_pos_return_2018=[]
    thursday_comb_return_2018=[]
    friday_all_neg_return_2018=[]
    friday_all_pos_return_2018=[]
    friday_comb_return_2018=[]

    for line in  data_line_2018[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2018.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2018.append(float(line[-3]))
            else:
                monday_all_neg_return_2018.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2018.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2018.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2018.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2018.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2018.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2018.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2018.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2018.append(float(line[-3]))
            else:
                thursday_all_neg_return_2018.append(float(line[-3]))        
        else :
            friday_comb_return_2018.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2018.append(float(line[-3]))
            else:
                friday_all_neg_return_2018.append(float(line[-3]))        
        
    ## Summarizing the result for 2018- Monday
    monday_comb_2018=['Monday']
    
    monday_r_neg_2018_mean = round(statistics.mean(monday_all_neg_return_2018),5)
    monday_r_neg_2018_std = round(statistics.stdev(monday_all_neg_return_2018),5)
    monday_r_neg_2018_cnt = len(monday_all_neg_return_2018)
    monday_r_pos_2018_mean = round(statistics.mean(monday_all_pos_return_2018),5)
    monday_r_pos_2018_std = round(statistics.stdev(monday_all_pos_return_2018),5)
    monday_r_pos_2018_cnt = len(monday_all_pos_return_2018)
    monday_r_comb_2018_mean = round(statistics.mean(monday_comb_return_2018),5)
    monday_r_comb_2018_std = round(statistics.stdev(monday_comb_return_2018),5)
    
    # Appendig the result to empty string for Monday 2018
    monday_comb_2018.append(monday_r_neg_2018_mean)
    monday_comb_2018.append(monday_r_neg_2018_std)
    monday_comb_2018.append(monday_r_neg_2018_cnt)
    monday_comb_2018.append(monday_r_pos_2018_mean)
    monday_comb_2018.append(monday_r_pos_2018_std)
    monday_comb_2018.append(monday_r_pos_2018_cnt)
    monday_comb_2018.append(monday_r_comb_2018_mean)
    monday_comb_2018.append(monday_r_comb_2018_std)

    ## Summarizing the result for 2018- tuesday
    tuesday_comb_2018=['Tuesday']
    
    tuesday_r_neg_2018_mean = round(statistics.mean(tuesday_all_neg_return_2018),5)
    tuesday_r_neg_2018_std = round(statistics.stdev(tuesday_all_neg_return_2018),5)
    tuesday_r_neg_2018_cnt = len(tuesday_all_neg_return_2018)
    tuesday_r_pos_2018_mean = round(statistics.mean(tuesday_all_pos_return_2018),5)
    tuesday_r_pos_2018_std = round(statistics.stdev(tuesday_all_pos_return_2018),5)
    tuesday_r_pos_2018_cnt = len(tuesday_all_pos_return_2018)
    tuesday_r_comb_2018_mean = round(statistics.mean(tuesday_comb_return_2018),5)
    tuesday_r_comb_2018_std = round(statistics.stdev(tuesday_comb_return_2018),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2018.append(tuesday_r_neg_2018_mean)
    tuesday_comb_2018.append(tuesday_r_neg_2018_std)
    tuesday_comb_2018.append(tuesday_r_neg_2018_cnt)
    tuesday_comb_2018.append(tuesday_r_pos_2018_mean)
    tuesday_comb_2018.append(tuesday_r_pos_2018_std)
    tuesday_comb_2018.append(tuesday_r_pos_2018_cnt)
    tuesday_comb_2018.append(tuesday_r_comb_2018_mean)
    tuesday_comb_2018.append(tuesday_r_comb_2018_std)    
    
    ## Summarizing the result for 2018- wednesday
    wednesday_comb_2018=['Wednesday']
    
    wednesday_r_neg_2018_mean = round(statistics.mean(wednesday_all_neg_return_2018),5)
    wednesday_r_neg_2018_std = round(statistics.stdev(wednesday_all_neg_return_2018),5)
    wednesday_r_neg_2018_cnt = len(wednesday_all_neg_return_2018)
    wednesday_r_pos_2018_mean = round(statistics.mean(wednesday_all_pos_return_2018),5)
    wednesday_r_pos_2018_std = round(statistics.stdev(wednesday_all_pos_return_2018),5)
    wednesday_r_pos_2018_cnt = len(wednesday_all_pos_return_2018)
    wednesday_r_comb_2018_mean = round(statistics.mean(wednesday_comb_return_2018),5)
    wednesday_r_comb_2018_std = round(statistics.stdev(wednesday_comb_return_2018),5)
    
    # Appendig the result to empty string for wednesday 2018
    wednesday_comb_2018.append(wednesday_r_neg_2018_mean)
    wednesday_comb_2018.append(wednesday_r_neg_2018_std)
    wednesday_comb_2018.append(wednesday_r_neg_2018_cnt)
    wednesday_comb_2018.append(wednesday_r_pos_2018_mean)
    wednesday_comb_2018.append(wednesday_r_pos_2018_std)
    wednesday_comb_2018.append(wednesday_r_pos_2018_cnt)
    wednesday_comb_2018.append(wednesday_r_comb_2018_mean)
    wednesday_comb_2018.append(wednesday_r_comb_2018_std)
    
    ## Summarizing the result for 2018- thursday
    thursday_comb_2018=['Thursday']
    
    thursday_r_neg_2018_mean = round(statistics.mean(thursday_all_neg_return_2018),5)
    thursday_r_neg_2018_std = round(statistics.stdev(thursday_all_neg_return_2018),5)
    thursday_r_neg_2018_cnt = len(thursday_all_neg_return_2018)
    thursday_r_pos_2018_mean = round(statistics.mean(thursday_all_pos_return_2018),5)
    thursday_r_pos_2018_std = round(statistics.stdev(thursday_all_pos_return_2018),5)
    thursday_r_pos_2018_cnt = len(thursday_all_pos_return_2018)
    thursday_r_comb_2018_mean = round(statistics.mean(thursday_comb_return_2018),5)
    thursday_r_comb_2018_std = round(statistics.stdev(thursday_comb_return_2018),5)
    
    # Appendig the result to empty string for thursday 2018
    thursday_comb_2018.append(thursday_r_neg_2018_mean)
    thursday_comb_2018.append(thursday_r_neg_2018_std)
    thursday_comb_2018.append(thursday_r_neg_2018_cnt)
    thursday_comb_2018.append(thursday_r_pos_2018_mean)
    thursday_comb_2018.append(thursday_r_pos_2018_std)
    thursday_comb_2018.append(thursday_r_pos_2018_cnt)
    thursday_comb_2018.append(thursday_r_comb_2018_mean)
    thursday_comb_2018.append(thursday_r_comb_2018_std)    

    ## Summarizing the result for 2018- friday
    friday_comb_2018=['Friday']
    
    friday_r_neg_2018_mean = round(statistics.mean(friday_all_neg_return_2018),5)
    friday_r_neg_2018_std = round(statistics.stdev(friday_all_neg_return_2018),5)
    friday_r_neg_2018_cnt = len(friday_all_neg_return_2018)
    friday_r_pos_2018_mean = round(statistics.mean(friday_all_pos_return_2018),5)
    friday_r_pos_2018_std = round(statistics.stdev(friday_all_pos_return_2018),5)
    friday_r_pos_2018_cnt = len(friday_all_pos_return_2018)
    friday_r_comb_2018_mean = round(statistics.mean(friday_comb_return_2018),5)
    friday_r_comb_2018_std = round(statistics.stdev(friday_comb_return_2018),5)
    
    # Appendig the result to empty string for friday 2018
    friday_comb_2018.append(friday_r_neg_2018_mean)
    friday_comb_2018.append(friday_r_neg_2018_std)
    friday_comb_2018.append(friday_r_neg_2018_cnt)
    friday_comb_2018.append(friday_r_pos_2018_mean)
    friday_comb_2018.append(friday_r_pos_2018_std)
    friday_comb_2018.append(friday_r_pos_2018_cnt)
    friday_comb_2018.append(friday_r_comb_2018_mean)
    friday_comb_2018.append(friday_r_comb_2018_std)
    
    ## Printing the table for 2018
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2018)
    t.add_row(tuesday_comb_2018)
    t.add_row(wednesday_comb_2018)
    t.add_row(thursday_comb_2018)
    t.add_row(friday_comb_2018)
    
    print('\n')
    print('Year 2018 \n')
    print(t)

   # Creating list for Daily return, Negative returns and pos return for 2019


    monday_all_neg_return_2019=[]
    monday_all_pos_return_2019=[]
    monday_comb_return_2019=[]
    tuesday_all_neg_return_2019=[]
    tuesday_all_pos_return_2019=[]
    tuesday_comb_return_2019=[]
    wednesday_all_neg_return_2019=[]
    wednesday_all_pos_return_2019=[]
    wednesday_comb_return_2019=[]
    thursday_all_neg_return_2019=[]
    thursday_all_pos_return_2019=[]
    thursday_comb_return_2019=[]
    friday_all_neg_return_2019=[]
    friday_all_pos_return_2019=[]
    friday_comb_return_2019=[]

    for line in  data_line_2019[:]:
        if line[4]== 'Monday' :
            monday_comb_return_2019.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                monday_all_pos_return_2019.append(float(line[-3]))
            else:
                monday_all_neg_return_2019.append(float(line[-3]))
        elif line[4]== 'Tuesday' :        
            tuesday_comb_return_2019.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                tuesday_all_pos_return_2019.append(float(line[-3]))
            else:
                tuesday_all_neg_return_2019.append(float(line[-3]))
        elif line[4]== 'Wednesday' :   
            wednesday_comb_return_2019.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                wednesday_all_pos_return_2019.append(float(line[-3]))
            else:
                wednesday_all_neg_return_2019.append(float(line[-3]))        
        elif line[4]== 'Thursday' : 
            thursday_comb_return_2019.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                thursday_all_pos_return_2019.append(float(line[-3]))
            else:
                thursday_all_neg_return_2019.append(float(line[-3]))        
        else :
            friday_comb_return_2019.append(float(line[-3]))
            if float(line[-3]) >= 0 :
                friday_all_pos_return_2019.append(float(line[-3]))
            else:
                friday_all_neg_return_2019.append(float(line[-3]))        
        
    ## Summarizing the result for 2019- Monday
    monday_comb_2019=['Monday']
    
    monday_r_neg_2019_mean = round(statistics.mean(monday_all_neg_return_2019),5)
    monday_r_neg_2019_std = round(statistics.stdev(monday_all_neg_return_2019),5)
    monday_r_neg_2019_cnt = len(monday_all_neg_return_2019)
    monday_r_pos_2019_mean = round(statistics.mean(monday_all_pos_return_2019),5)
    monday_r_pos_2019_std = round(statistics.stdev(monday_all_pos_return_2019),5)
    monday_r_pos_2019_cnt = len(monday_all_pos_return_2019)
    monday_r_comb_2019_mean = round(statistics.mean(monday_comb_return_2019),5)
    monday_r_comb_2019_std = round(statistics.stdev(monday_comb_return_2019),5)
    
    # Appendig the result to empty string for Monday 2019
    monday_comb_2019.append(monday_r_neg_2019_mean)
    monday_comb_2019.append(monday_r_neg_2019_std)
    monday_comb_2019.append(monday_r_neg_2019_cnt)
    monday_comb_2019.append(monday_r_pos_2019_mean)
    monday_comb_2019.append(monday_r_pos_2019_std)
    monday_comb_2019.append(monday_r_pos_2019_cnt)
    monday_comb_2019.append(monday_r_comb_2019_mean)
    monday_comb_2019.append(monday_r_comb_2019_std)

    ## Summarizing the result for 2019- tuesday
    tuesday_comb_2019=['Tuesday']
    
    tuesday_r_neg_2019_mean = round(statistics.mean(tuesday_all_neg_return_2019),5)
    tuesday_r_neg_2019_std = round(statistics.stdev(tuesday_all_neg_return_2019),5)
    tuesday_r_neg_2019_cnt = len(tuesday_all_neg_return_2019)
    tuesday_r_pos_2019_mean = round(statistics.mean(tuesday_all_pos_return_2019),5)
    tuesday_r_pos_2019_std = round(statistics.stdev(tuesday_all_pos_return_2019),5)
    tuesday_r_pos_2019_cnt = len(tuesday_all_pos_return_2019)
    tuesday_r_comb_2019_mean = round(statistics.mean(tuesday_comb_return_2019),5)
    tuesday_r_comb_2019_std = round(statistics.stdev(tuesday_comb_return_2019),5)
    
    # Appendig the result to empty string for tuesday 2014
    tuesday_comb_2019.append(tuesday_r_neg_2019_mean)
    tuesday_comb_2019.append(tuesday_r_neg_2019_std)
    tuesday_comb_2019.append(tuesday_r_neg_2019_cnt)
    tuesday_comb_2019.append(tuesday_r_pos_2019_mean)
    tuesday_comb_2019.append(tuesday_r_pos_2019_std)
    tuesday_comb_2019.append(tuesday_r_pos_2019_cnt)
    tuesday_comb_2019.append(tuesday_r_comb_2019_mean)
    tuesday_comb_2019.append(tuesday_r_comb_2019_std)    
    
    ## Summarizing the result for 2019- wednesday
    wednesday_comb_2019=['Wednesday']
    
    wednesday_r_neg_2019_mean = round(statistics.mean(wednesday_all_neg_return_2019),5)
    wednesday_r_neg_2019_std = round(statistics.stdev(wednesday_all_neg_return_2019),5)
    wednesday_r_neg_2019_cnt = len(wednesday_all_neg_return_2019)
    wednesday_r_pos_2019_mean = round(statistics.mean(wednesday_all_pos_return_2019),5)
    wednesday_r_pos_2019_std = round(statistics.stdev(wednesday_all_pos_return_2019),5)
    wednesday_r_pos_2019_cnt = len(wednesday_all_pos_return_2019)
    wednesday_r_comb_2019_mean = round(statistics.mean(wednesday_comb_return_2019),5)
    wednesday_r_comb_2019_std = round(statistics.stdev(wednesday_comb_return_2019),5)
    
    # Appendig the result to empty string for wednesday 2019
    wednesday_comb_2019.append(wednesday_r_neg_2019_mean)
    wednesday_comb_2019.append(wednesday_r_neg_2019_std)
    wednesday_comb_2019.append(wednesday_r_neg_2019_cnt)
    wednesday_comb_2019.append(wednesday_r_pos_2019_mean)
    wednesday_comb_2019.append(wednesday_r_pos_2019_std)
    wednesday_comb_2019.append(wednesday_r_pos_2019_cnt)
    wednesday_comb_2019.append(wednesday_r_comb_2019_mean)
    wednesday_comb_2019.append(wednesday_r_comb_2019_std)
    
    ## Summarizing the result for 2019- thursday
    thursday_comb_2019=['Thursday']
    
    thursday_r_neg_2019_mean = round(statistics.mean(thursday_all_neg_return_2019),5)
    thursday_r_neg_2019_std = round(statistics.stdev(thursday_all_neg_return_2019),5)
    thursday_r_neg_2019_cnt = len(thursday_all_neg_return_2019)
    thursday_r_pos_2019_mean = round(statistics.mean(thursday_all_pos_return_2019),5)
    thursday_r_pos_2019_std = round(statistics.stdev(thursday_all_pos_return_2019),5)
    thursday_r_pos_2019_cnt = len(thursday_all_pos_return_2019)
    thursday_r_comb_2019_mean = round(statistics.mean(thursday_comb_return_2019),5)
    thursday_r_comb_2019_std = round(statistics.stdev(thursday_comb_return_2019),5)
    
    # Appendig the result to empty string for thursday 2019
    thursday_comb_2019.append(thursday_r_neg_2019_mean)
    thursday_comb_2019.append(thursday_r_neg_2019_std)
    thursday_comb_2019.append(thursday_r_neg_2019_cnt)
    thursday_comb_2019.append(thursday_r_pos_2019_mean)
    thursday_comb_2019.append(thursday_r_pos_2019_std)
    thursday_comb_2019.append(thursday_r_pos_2019_cnt)
    thursday_comb_2019.append(thursday_r_comb_2019_mean)
    thursday_comb_2019.append(thursday_r_comb_2019_std)    

    ## Summarizing the result for 2019- friday
    friday_comb_2019=['Friday']
    
    friday_r_neg_2019_mean = round(statistics.mean(friday_all_neg_return_2019),5)
    friday_r_neg_2019_std = round(statistics.stdev(friday_all_neg_return_2019),5)
    friday_r_neg_2019_cnt = len(friday_all_neg_return_2019)
    friday_r_pos_2019_mean = round(statistics.mean(friday_all_pos_return_2019),5)
    friday_r_pos_2019_std = round(statistics.stdev(friday_all_pos_return_2019),5)
    friday_r_pos_2019_cnt = len(friday_all_pos_return_2019)
    friday_r_comb_2019_mean = round(statistics.mean(friday_comb_return_2019),5)
    friday_r_comb_2019_std = round(statistics.stdev(friday_comb_return_2019),5)
    
    # Appendig the result to empty string for friday 2019
    friday_comb_2019.append(friday_r_neg_2019_mean)
    friday_comb_2019.append(friday_r_neg_2019_std)
    friday_comb_2019.append(friday_r_neg_2019_cnt)
    friday_comb_2019.append(friday_r_pos_2019_mean)
    friday_comb_2019.append(friday_r_pos_2019_std)
    friday_comb_2019.append(friday_r_pos_2019_cnt)
    friday_comb_2019.append(friday_r_comb_2019_mean)
    friday_comb_2019.append(friday_r_comb_2019_std)
    
    ## Printing the table for 2019
    

    t = PrettyTable(['Day','Mu(R)', 'std(R)','|(R-)|','Mu(R-)', 'std(R-)','|(R+)|','Mu(R+)', 'std(R+)'])
    t.add_row(monday_comb_2019)
    t.add_row(tuesday_comb_2019)
    t.add_row(wednesday_comb_2019)
    t.add_row(thursday_comb_2019)
    t.add_row(friday_comb_2019)
    
    print('\n')
    print('Year 2019 \n')
    print(t)
    
    print('\n')
    print("Question 1 Part 3. are there more days with negative or non-negative returns?")
    print("Answer- There is no clear pattern but looking at all the years it seems it toggles in different years")
    
    print('\n')
    print("Question 1 Part 4. does your stock lose more on a DOWN day than it gains on an UP days.")
    print("Answer- Again thre is not cleaer pattern to establish this. I believe it varies but if I pick one sample for e..g in 2015 the stock lose more on down days than gain in the up days")
    
    print('\n')
    print("Question 1 Part 5. are these results the same across days of the week?")
    print("Answer-  The results are not same across different days, each day have different pattern")
    print('\n')
    print('*******************')      
    print('\n')
    print("Question 2 Part 1. are there any patterns across days of the week?")
    print("Answer-  I do not see any pattern at all, I may be wrong but I do not see any pattern whatso ever")
          
    print('\n')
    print("Question 2 Part 2. are there any patterns across dierent years for the sameday of the week?")
    print("Answer-Looking at data across the years, Wednesday are mostly postive while Thursday are negative, Fridays are just hovering around to be postive in general") 
    
    print('\n')
    print("Question 2 Part 3.what are the best and worst days of the week to be investedfor each year.")
    print("Answer-Per For 2015 my observation the best day of the week is Tuesday while the worst day of the week is Friday")
    print("Answer-Per For 2016 my observation the best day of the week is Tuesday  while the worst day of the week is Friday")
    print("Answer-Per For 2017 my observation the best day of the week is Monday  while the worst day of the week is Thursday")
    print("Answer-Per For 2018 my observation the best day of the week is Wednesday  while the worst day of the week is Tuesday")
    print("Answer-Per For 2019 my observation the best day of the week is Thursday  while the worst day of the week is Tuesday")
    
    print('\n')
    print("Question 2 Part 4.do these days change from year to year for your stock?.")
    print("Answer- The days do vary with each year, it is not a consistent pattern")      
          
          
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)












