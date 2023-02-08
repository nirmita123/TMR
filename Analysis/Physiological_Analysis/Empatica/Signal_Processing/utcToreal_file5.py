import pandas as pd
from datetime import datetime as pt
import numpy as np
import datetime as dt
from pandas.errors import EmptyDataError


def read_data(date):
    """
    Argument date: file from while it reads HRV, EDA and HR
    returns data frame (pandas object) for each HRV, EDA, HR and tags files
    """
    dfHRV = pd.read_csv(base+date+'/HRV.csv')
    dfEDA = pd.read_csv(base+date+'/newEDA.csv')
    dfHR = pd.read_csv(base+date+'/newHR.csv')
    try: # checks if the tags file is empty
        dfTags = pd.read_csv(base+date+'/tags.csv')
    except EmptyDataError:
        dfTags = pd.DataFrame()

    return dfHRV, dfEDA, dfHR, dfTags


def convert_time(unix):
    """
    Argument unix: unix timestamp
    returns human readable timestamp (adjusted according to Local time zone of Boston(EST))
    """
    return (pt.utcfromtimestamp(unix) + dt.timedelta(0,0,0,-875,0,-5,0))

def add_tags(base,date,df4):
    """
    Argument base: base of the file
    returns human readable timestamp (adjusted according to Local time zone of Boston(EST))
    """
    df4 = pd.read_csv(base+date+'/tags.csv', usecols=[0], names=['colA'], header=None)

    for val in range(len(df4['colA'])):
        df4['colA'][val]=(pt.utcfromtimestamp(df4['colA'][val]) + dt.timedelta(0,0,0,-875,0,-5,0) ).strftime('%H:%M:%S')

    df4.to_csv(base+date+'/tags_1.csv', index=False)

def convert_file(date_rng,df0,file,date,typ):
    """
    Argument date_rng: list of dates (timestamps)
    df0: dataframe to read the data
    date: subject data of one session
    file: file name to keep of the new file
    typ: type of data (HR,HRV,EDA)
        
    returns human readable timestamp (adjusted according to Local time zone of Boston(EST))
    """
    df = pd.DataFrame(date_rng.strftime('%H:%M:%S'), columns=['Timestamp'])
    df[typ] = df0[typ]
    df.head(15)
    df.to_csv(base+date+"/"+file+".csv", index=False)

baseAwake="../../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
baseSleep="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"

base=baseAwake

# file names of wake data
# Date=["2_control","3_control","5_control","6_control","16_control","14_control","11_control","7_control","2_scent","3_scent","5_scent","6_scent","16_scent","14_scent","11_scent","7_scent"]
Date=["17_control","17_scent","19_control","19_scent","20_control","20_scent","22_control","22_scent","23_control","23_scent","24_control","24_scent","25_control","25_scent","26_control","26_scent","27_control","27_scent","28_control","28_scent","29_control","29_scent","30_control","30_scent","31_control","31_scent","32_control","32_scent"]

# file names of sleep data
# Date=["2_c","2_s","3_c","3_s","4_c","4_s","5_c","5_s","6_c","6_s"]
# Date=["25_control"]
# Date = ["17_control","17_scent","18_control","18_scent","20_control","20_scent","25_control","25_scent","28_control","28_scent","30_control","30_scent"]
for date1 in Date:
    
    dfHRV, dfEDA, dfHR, dfTags = read_data(date1)
         
    start_unix=dfHRV['Timestamp'].iloc[0] #read start unix timestamp from HRV
    end_unix=dfHRV['Timestamp'].iloc[-1] #read end unix timestamp from HRV
    
    # convert start and end unix timestamp to human readable form
    start_human = convert_time(start_unix)
    end_human = convert_time(end_unix)
    
    # calculate frequency of the HRV file
    f= (end_unix-start_unix)/len(dfHRV)
    # frequency for HR files = 1Hz and EDA files = 4Hz -> constant for all
    print(f)
    # get the list of time, according to the frequency, for HRV, EDA and HR
    # For eg.: date_rngEDA=[start_human, start_human+0.25, start_human+0.5,..,end_human]
    date_rngHRV = pd.date_range(start=start_human.strftime('%H:%M:%S'), end=end_human.strftime('%H:%M:%S'), freq=str(int(f*1000))+"ms")
    date_rngEDA = pd.date_range(start=start_human.strftime('%H:%M:%S'), end=end_human.strftime('%H:%M:%S'), freq="250ms")
    date_rngHR = pd.date_range(start=start_human.strftime('%H:%M:%S'), end=end_human.strftime('%H:%M:%S'), freq="1s")

    # creates new file with human readable timestamps
    convert_file(date_rngHRV,dfHRV,"HRV_main",date1,"HRV")
    convert_file(date_rngEDA,dfEDA,"EDA_main",date1,"EDA")
    convert_file(date_rngHR,dfHR,"HR_main",date1,"HR")
    
    # creates new file with human readable timestamps for tags
    add_tags(base,date1,dfTags)
    print(date1+" done")


