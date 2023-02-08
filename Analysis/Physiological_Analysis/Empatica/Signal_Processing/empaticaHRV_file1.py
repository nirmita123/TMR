

#########################################################################
#                                                                       #
#                       Jesus Garcia-Mancilla                           #
#                            Scrapworks                                 #
#                           October 2017                                #
#                                                                       #
#########################################################################

import pandas as pd
import numpy as np
import peakutils
import math, sys
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.signal import hilbert

print('\nStart')
def process(date):
    BVP_DF = pd.read_csv(base+date+'/BVP.csv')
    HR_DF = pd.read_csv(base+date+'/HR.csv')

    column = list(HR_DF)[0]
    temp = HR_DF.drop(0, axis = 0)
    HR = temp[column]
    HR = HR.tolist()

    column2 = list(BVP_DF)[0]
    sample_rate = BVP_DF[column2][0]
    temp = BVP_DF.drop(0, axis = 0)
    temp['spData'] = 0
    temp.loc[temp[column2] > 0, 'spData'] = temp[column2]
    signal = temp['spData'].tolist()

    return signal,column2,sample_rate,HR

def bvpPeaks(signal):
    cb = np.array(signal)
    x = peakutils.indexes(cb, thres=0.02/max(cb), min_dist=0.1)
    y = []
    i = 0
    while (i < (len(x)-1)):
        if x[i+1] - x[i] < 15:
            y.append(x[i])
            x = np.delete(x, i+1)
        else:
            y.append(x[i])
        i += 1
    return y

def getRRI(signal, start, sample_rate):
    peakIDX = bvpPeaks(signal)
    spr = 1 / sample_rate # seconds between readings
    start_time = float(start)
    timestamp = [start_time, (peakIDX[0] * spr) + start_time ]
    ibi = [0, 0]
    for i in range(1, len(peakIDX)):
        timestamp.append(peakIDX[i] * spr + start_time)
        ibi.append((peakIDX[i] - peakIDX[i-1]) * spr)

    df = pd.DataFrame({'Timestamp': timestamp, 'IBI': ibi})
    return df

def getHRV(data, avg_heart_rate, date):
    rri = pd.read_csv(base+date+'/IBI.csv')
    column = list(rri)[0]
    temp = rri.drop(0, axis = 0)
    rri = temp[column]
    RR_list = rri.tolist()
    RR_sqdiff = []

    RR_diff_timestamp = []
    cnt = 2
    while (cnt < (len(RR_list)-1)):
        RR_sqdiff.append(math.pow(RR_list[cnt+1] - RR_list[cnt], 2))
        RR_diff_timestamp.append(data['Timestamp'][cnt])
        cnt += 1
    hrv_window_length = 10
    window_length_samples = int(hrv_window_length*(avg_heart_rate/60))
    RMSSD = []
    Ab_rmssd= []
    index = 1
    for val in RR_sqdiff:
        if index < int(window_length_samples):
            RMSSDchunk = RR_sqdiff[:index:]
        else:
            RMSSDchunk = RR_sqdiff[(index-window_length_samples):index:]
        RMSSD.append(math.sqrt(np.std(RMSSDchunk)))
        Ab_rmssd.append(abs(math.sqrt(np.std(RMSSDchunk))))

        index += 1
    #dt = np.dtype('Float64')
    RMSSD = np.array(RMSSD)
    Ab_rmssd = np.array(Ab_rmssd)

    signal=RMSSD
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)

    df = pd.DataFrame({'Timestamp': RR_diff_timestamp, 'HRV': RMSSD})
#     df1 = df.rolling(window=250).mean()
#     df2 = pd.DataFrame({'Timestamp': RR_diff_timestamp, 'HRV': amplitude_envelope})

    return (df,RR_diff_timestamp)

def Average(lst):
    return sum(lst) / len(lst)

def send2csv(date,HRV_DF):
    HRV_DF.to_csv(base+date+'/HRV.csv', index=False)

#     HRV_M.to_csv(base+date+'/HRV_M.csv', index=False)
#     ABS_RMSSD.to_csv(base+date+'/HRV_Abs.csv', index=False)

    print('\n    Done, saved as: HRV.csv\n')



baseAwake="../../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
baseSleep="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"


base=baseAwake

DateAwake=["17_control","17_scent","19_control","19_scent","20_control","20_scent","22_control","22_scent_1","22_scent_2","23_control","23_scent","24_control","24_scent","25_control","25_scent","26_control","26_scent","27_control","27_scent","28_control","28_scent","29_control","29_scent","30_control","30_scent","31_control","31_scent","32_control","32_scent"]
#DateAwake=["20_control_r","28_scent_r"]
Date=DateAwake

for i in range(len(Date)):
    date1=Date[i]
    print(Date[i])
    signal,column2,sample_rate,HR= process(date1)

    RRI_DF = getRRI(signal, column2, sample_rate)
    HRV_DF, t= getHRV(RRI_DF, np.mean(HR), date1)
    send2csv(date1,HRV_DF)


