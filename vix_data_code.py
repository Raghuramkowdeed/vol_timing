#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:30:21 2018

@author: raghuramkowdeed
"""

import datetime as dt
from pandas import *
import os
import urllib2
import dateutil.relativedelta as rd
#from csvDatabase import HistDataCsv

m_codes = dict(zip(range(1,13),['F','G','H','J','K','M','N','Q','U','V','X','Z'])) #month codes of the futures
monthToCode = dict(zip(range(1,len(m_codes)+1),m_codes))


def getCboeData(year,month):

    ''' download data from cboe '''
    fName = "CFE_{0}{1}_VX.csv".format(m_codes[month],str(year)[-2:])
    urlStr =  "http://cfe.cboe.com/Publish/ScheduledTask/MktData/datahouse/{0}".format(fName)

    try:
        lines = urllib2.urlopen(urlStr).readlines()
    except Exception as e:
        print('error')
        #s = "Failed to download:\n{0}".format(e);
        #print s

    # first column is date, second is future , skip these


    line_num= 0
    stop_proc = False

    while not stop_proc :
        this_line = lines[line_num]
        try :
            fields = this_line.strip().split(',')
            this_date = datetime.strptime( fields[0],'%m/%d/%Y')
            stop_proc = True
        except :
            line_num = line_num + 1

    header = lines[line_num-1].strip().split(',')[2:]

    dates = []    
    data = [[] for i in range(len(header))]



    for line in lines[line_num:]:
        fields = line.strip().split(',')
        dates.append(datetime.strptime( fields[0],'%m/%d/%Y'))
        for i,field in enumerate(fields[2:]):
            data[i].append(float(field))

    data = dict(zip(header,data))   

    df = DataFrame(data=data, index=Index(dates))

    return df

    
