# filter data using NC-VT (Non-causal of Variable Threshold filter)
import pandas as pd
import statistics as stt

baseAwake="../../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
baseSleep="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"


base=baseSleep #change it to baseSleep for sleep data

# Awake files
# Date=["2_control","3_control","5_control","6_control","16_control","14_control","11_control","7_control","2_scent","3_scent","5_scent","6_scent","16_scent","14_scent","11_scent","7_scent"]
# Date=["17_control","17_scent","19_control","19_scent","20_control","20_scent","22_control","22_scent_1","22_scent_2","23_control","23_scent","24_control","24_scent","25_control","25_scent","26_control","26_scent","27_control","27_scent","28_control","28_scent","29_control","29_scent","30_control","30_scent","31_control","31_scent","32_control","32_scent"]

# sleep
# Date=["2_c_1","2_c_2","2_c_r","2_s","3_c","3_s","4_c","4_s","5_c","5_s","5_s_r","6_c","6_s","6_s_r"]
Date = ["17_control","17_scent","18_control","18_scent","20_control","20_scent","25_control","25_scent","28_control","28_scent","30_control","30_scent"]


def NCVT(j,st,file):
    # NC-VT
    # Read the related file i.e. HR/HRV/EDA
    old_data=pd.read_csv(base+Date[j]+"/"+file+".csv")
    
    # Filtered list of the old file
    new_data=[]
    t=[]
    if(len(old_data)!=0):
        M=stt.mean(old_data[st]) # mean kf old data

        # u1 is +-30% of old_data[i-1]
        u1= old_data[st][0]*0.3

        # u2 is 30% of Mean
        u2= M*0.3

        c=0
        c1=0
        

        for i in range(1, len(old_data)-1):
            t.append(old_data['Timestamp'][i])
            if abs(old_data[st][i]-old_data[st][len(old_data)-1])/old_data[st][len(old_data)-1] < u1 or abs(old_data[st][i]-old_data[st][i+1])/old_data[st][len(old_data)-1]< u1 or abs(old_data[st][i]-M)/M <u2:
                #accept data element
                new_data.append(old_data[st][i])
                #update u1 and u2
                u1=old_data[st][i-1]*0.3
                c1=c1+1 #counter
            else:
                c=c+1 #counter
                #add the data as the average of mean and previous value
                new_data.append((old_data[st][i-1]+M)/2)

                i=i+1
        print("c"+str(c))
        print("c1"+str(c1))
    # creating final filtered file
    new = pd.DataFrame({'Timestamp': t, st: new_data})
    # saving it as csv
    new.to_csv(base+Date[j]+"/new"+file+".csv")
    # print the count of the data that has been changed
    


# files to be filtered
file=["HR","EDA","HRV"]
# ,"HR_recall","EDA_recall","HRV_recall"
sts=["HR","EDA","HRV"]
# ,"HR","EDA","HRV"
for i in range(len(Date)):
    print(Date[i])
    for j in range(len(sts)):
        NCVT(i,sts[j],file[j])



