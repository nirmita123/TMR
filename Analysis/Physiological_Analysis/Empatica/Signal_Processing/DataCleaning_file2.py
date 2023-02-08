
import time
import pandas as pd
import datetime
from statistics import mean


baseAwake="../../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
baseSleep="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"

base=baseSleep

def cleanDay(date,t):

    print(date)
#read EDA, HR and HRV
    d= pd.read_csv(base+date+"/EDA.csv", header=None)
    d1= pd.read_csv(base+date+"/HR.csv", header=None)
    d2= pd.read_csv(base+date+"/HRV.csv")

#read Timestamp (t)
    start_time=d[0][0]
    st=d1[0][0]
    st1=d2['Timestamp'][0]
    print(start_time,st,st1)
    
    timestampEDA = []
    timestampHR = []
    timestampHRV = []

    newEDA=[]
    newHR=[]
    newHRV=[]

# Saves the data collected in the first 2 minutes of the study --> it can be used as a baseline measure
    baselineEDA=[]
    baselineHR=[]
    baselineHRV=[]

# iterates over the EDA file appends the data (timestamp and EDA) to a new list(newEDA) until it reaches the timestamp
    for i in range(2,len(d)):
        x= (i * 0.25 + start_time )
        if i<120*4:
            baselineEDA.append(d[0][i])
        if (x<t):
            timestampEDA.append(i * 0.25 + start_time)
            newEDA.append(d[0][i])
            
# iterates over the HR file appends the data (timestamp and HR) to a new list(newHR) until it reaches the timestamp
    for i in range(2,len(d1)):
        x=(i + st)
        if i<120:
            baselineHR.append(d1[0][i])
        if (x<t) :
            timestampHR.append(i + st)
            newHR.append(d1[0][i])

# iterates over the HRV file appends the data (timestamp and HRV) to a new list(newHRV) until it reaches the timestamp
    for i in range(2,len(d2)):
        x=(i + st)
        if i<120:
            baselineHRV.append(d2['HRV'][i])
        if (d2['Timestamp'][i]<t) :
            timestampHRV.append(d2['Timestamp'][i])
            newHRV.append(d2['HRV'][i])

#make pandas object with Timestamp + EDA/HR/HRV
    dfEDA = pd.DataFrame({'Timestamp': timestampEDA, 'EDA': newEDA})
    dfHR = pd.DataFrame({'Timestamp': timestampHR, 'HR': newHR})
    dfHRV = pd.DataFrame({'Timestamp': timestampHRV, 'HRV': newHRV})

    dfEDA.to_csv(base+date+'/EDA.csv', index=False)
    dfHR.to_csv(base+date+'/HR.csv', index=False)
    dfHRV.to_csv(base+date+'/HRV.csv', index=False)

    print("meanEDA: "+str(dfEDA['EDA'].mean()))
    print("baselineEDA: "+str(mean(baselineEDA)))
    print("meanHR: "+str(dfHR['HR'].mean()))
    print("baselineHR: "+str(mean(baselineHR)))
    print("meanHRV: "+str(dfHRV['HRV'].mean()))
    print("baselineHRV: "+str(mean(baselineHRV)))

# Date contains all the File names the data of which you want to chop
# Date=["17_control","17_scent","19_control","19_scent","20_control","20_scent","22_control","22_scent_1","22_scent_2","23_control","23_scent","24_control","24_scent","25_control","25_scent","26_control","26_scent","27_control","27_scent","28_control","28_scent","29_control","29_scent","30_control","30_scent","31_control","31_scent","32_control","32_scent"]
# Date=["17_control","17_scent","18_control","18_scent","20_control","20_scent","25_control","25_scent","28_control","28_scent","30_control","30_scent"]
Date=["20_control_r"]
t=[1579702154]
# sleep
# t=[1579635067,1580244507,1579627456,1580240811,1579720132,1580324555,1579975377,1580583720,1580072159,1580678520,1580157944,1580764380]
# t is a list of timestamps, when the user fills the survey, and the data beyong which is only noise
# wake
# t=[1579556936, 1580161492, 1579643934, 1580248136, 1579647697, 1580251986, 1579738162, 1580342282, 1580342282,1580423283, 1579816727, 1580420469, 1579821094, 1580506807, 1579902927, 1580511518, 1579908101, 1579913128, 1580518580, 1580593162, 1579989342, 1580596531, 1579992848, 1580679026, 1580076044, 1580079234, 1580684454, 1580687779, 1580083103]

for i in range(len(Date)):
    cleanDay(Date[i],t[i])

"""
baseAwake="../../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
baseSleep="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"


base=baseSleep

def cleanSleep(date,t1,t2,t3):

    print(date)
#read EDA, HR and HRV
    d= pd.read_csv(base+date+"/EDA.csv", header=None)
    d1= pd.read_csv(base+date+"/HR.csv", header=None)
    d2= pd.read_csv(base+date+"/HRV.csv")

#read Timestamp (t)
    start_time=float(d[0][0])
    st=float(d1[0][0])
    st1=float(d2['Timestamp'][0])
    print(start_time,st,st1)
    
    timestampEDA = []
    timestampHR = []
    timestampHRV = []

    newEDA=[]
    newHR=[]
    newHRV=[]

    timestampEDA_recall = []
    timestampHR_recall = []
    timestampHRV_recall = []

    newEDA_recall=[]
    newHR_recall=[]
    newHRV_recall=[]

# Saves the data collected in the first 2 minutes of the study --> it can be used as a baseline measure
    baselineEDA=[]
    baselineHR=[]
    baselineHRV=[]

# iterates over the EDA file appends the data (timestamp and EDA) to a new list(newEDA) until it reaches the timestamp
    for i in range(2,len(d)):
        x= (i * 0.25 + start_time )
        if i<120*4:
            baselineEDA.append(d[0][i])
        if (x>t1 and x<t2):
            timestampEDA.append(x)
            newEDA.append(d[0][i])
        elif (x>t2 and x<t3):
            timestampEDA_recall.append(x)
            newEDA_recall.append(d[0][i])
            
# iterates over the HR file appends the data (timestamp and HR) to a new list(newHR) until it reaches the timestamp
    for i in range(2,len(d1)):
        x=(i + st)
        if i<120:
            baselineHR.append(d1[0][i])
        if (x>t1 and x<t2) :
            timestampHR.append(x)
            newHR.append(d1[0][i])
        elif (x>t2 and x<t3):
            timestampHR_recall.append(x)
            newHR_recall.append(d1[0][i])

# iterates over the HRV file appends the data (timestamp and HRV) to a new list(newHRV) until it reaches the timestamp
    for i in range(2,len(d2)):
        x=d2['Timestamp'][i]
        if i<120:
            baselineHRV.append(d2['HRV'][i])
        if (x>t1 and x<t2) :
            timestampHRV.append(x)
            newHRV.append(d2['HRV'][i])
        elif (x>t2 and x<t3):
            timestampHRV_recall.append(x)
            newHRV_recall.append(d2['HRV'][i])

#make pandas object with Timestamp + EDA/HR/HRV
    dfEDA = pd.DataFrame({'Timestamp': timestampEDA, 'EDA': newEDA})
    dfHR = pd.DataFrame({'Timestamp': timestampHR, 'HR': newHR})
    dfHRV = pd.DataFrame({'Timestamp': timestampHRV, 'HRV': newHRV})

    dfEDA.to_csv(base+date+'/EDA.csv', index=False)
    dfHR.to_csv(base+date+'/HR.csv', index=False)
    dfHRV.to_csv(base+date+'/HRV.csv', index=False)

#make pandas object for Recall files
    dfEDA_r = pd.DataFrame({'Timestamp': timestampEDA_recall, 'EDA': newEDA_recall})
    dfHR_r = pd.DataFrame({'Timestamp': timestampHR_recall, 'HR': newHR_recall})
    dfHRV_r = pd.DataFrame({'Timestamp': timestampHRV_recall, 'HRV': newHRV_recall})

    dfEDA_r.to_csv(base+date+'/EDA_recall.csv', index=False)
    dfHR_r.to_csv(base+date+'/HR_recall.csv', index=False)
    dfHRV_r.to_csv(base+date+'/HRV_recall.csv', index=False)

    print("meanEDA: "+str(dfEDA['EDA'].mean()))
    print("baselineEDA: "+str(mean(baselineEDA)))
    print("meanHR: "+str(dfHR['HR'].mean()))
    print("baselineHR: "+str(mean(baselineHR)))
    print("meanHRV: "+str(dfHRV['HRV'].mean()))
    print("baselineHRV: "+str(mean(baselineHRV)))

# Date contains all the File names the data of which you want to chop

# DateSleep=["2_c_1","2_c_2","2_s","3_c","3_s","4_c","4_s","5_c","5_s","6_c","6_s"]
# DateSleep=["17_control","18_control","20_control","25_control","28_control","30_control","17_scent","18_scent","20_scent","25_scent","28_scent","30_scent"]
DateSleep=["25_control","25_scent"]
Date=DateSleep
# t is a list of timestamps, when the user fills the survey, and the data beyong which is only noise

# t1=[1560144300,1560144300,1560750120,1560226770,1560827760,1560220980,1560825780,1560315600,1560920460,1560317100,1560927360]
# t2=[1560176340,1560176340,1560775740,1560252510,1560858000,1560239520,1560853620,1560341160,1560953820,1560344100,1560951540]
# t3=[1560181783,1560181783,1560781763,1560255791,1560859819,1560247796,1560855994,1560342959,1560957514,1560351148,1560954068]
t1=[1580542530,1579928160]
t2=[1580565090,1579956296]
t3=[1580565826,1579957350]
# t1=[1579577580,1579573860,1579673410,1580542530,1580630490,1580718360,1580196180,1580190600,1580279784,1579928160,1580028300,1580111580]
# t2=[1579616054,1579592753,1579700534,1580565090,1580654250,1580742235,1580225273,1580221696,1580304882,1579956296,1580052570,1580139196]
# t3=[1579617024,1579593233,1579702112,1580565826,1580655060,1580742714,1580226244,1580222737,1580306632,1579957350,1580053345,1580139922]

for i in range(len(Date)):
    cleanSleep(Date[i],t1[i],t2[i],t3[i])


"""