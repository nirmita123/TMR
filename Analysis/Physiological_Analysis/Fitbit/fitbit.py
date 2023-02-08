import numpy as np
from astropy.table import Table, Column
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats
import pylab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import nbimporter
import re

baseFitbit="../../../Data/Physiological_Data/Raw_Data/Fitbit/"
plots="../../../Plots/Physiological_Plots/Fitbit/HR/"


def star_col(Statistic,Pval):
    """
     Arguments:
        Statistic (float): statistic from the t-test
        Pval (float): the p-value of the data
    Output:
        star (string): could be "*","**" or "***" depending on the significance of the pvalue
        col (string): could be "k" or "r", where "k" is for black when scent is significant and "r" is for red when control is significant
    """
    star=''
    col='k'

    if(Pval<0.0001):
        star='***'
    elif Pval<0.001:
        star='**'
    elif Pval<0.05:
        star='*'
    return star,col

def compress(data):
    """
     Arguments:
        data (list): the list which needs to be compressed
    Output:
        x (list, size = 30): returns 30 data points(means) representing the list
    """
    x=[]
    y=int(len(data)/30)
    for j in range(0,(y*30),y):
        x.append(st.mean(data[j:j+y]))
        
#     print((x))
    return x
    
    
base2=baseFitbit
def star_col(Statistic,Pval):
    """
     Arguments:
        s (string): could be "HRV","HR" or "EDA
        Statistic (float): statistic from the t-test
        Pval (float): the p-value of the data
    Output:
        star (string): could be "*","**" or "***" depending on the significance of the pvalue
        col (string): could be "k" or "r", where "k" is for black when scent is significant and "r" is for red when control is significant
    """
    star=''
    col='k'

    if(Pval<0.0001):
        star='***'
    elif Pval<0.001:
        star='**'
    elif Pval<0.05:
        star='*'
    return star,col

def findIndex(t, st,study):#
    """
     Arguments:
        t (list): timestamp list of the data
        st (string): timestamp to find in the list
        study (int): could be 1 or 2, where 1 represents "study 1" and 2 represents "study 2"
    Output:
        returns index of the timestamp in the list
        
    """
    if(st[1]==':') and study == 1:
        st='0'+st

    for i in range(len(t)):
        if st[0]== t[i][0] and st[1] ==t[i][1] and st[2]==t[i][2] and st[3]==t[i][3]:
#             print("Start time found: "+str(t[i]))
            return i

            
    print("Start time not found")
    return -1
    
def fitbitMeanHR(startDate, endDate, startTime, endTime, subjectID, Folder, study):
    """
     Arguments:
        startDate (string): start date of sleep
        endDate(string): end date of sleep
        startTime(string): start time of sleep
        endTime(string): end time of sleep
        subjectID (int): could be from 0-31
        Folder (string): Folder could be Pixel, Samsung etc. representing the set of devices the subject used
        study (int): could be 1 or 2; 1 if study 1(first 16) and 2 if study 2(next 16)
    Output:
        x (list): containing all the data of sleep of one day of that subject
    """
    global base2
    m=[]
    x=[]
    c=0
#    if study == 2:
#            base2=base2
            
    for i in range(len(startDate)):
        if(startDate[i]==endDate[i]):
            
#             read start date file
            
            df=pd.read_csv(base2+Folder[i]+'/'+startDate[i]+'.csv')

            index1=findIndex(df['Time'],startTime[i],study)
            index2=findIndex(df['Time'],endTime[i],study)
            if index1==-1 or index2==-1:
                m.append(-1)
                print(Folder[i],startDate[i],startTime[i],endTime[i],index1,index2)
                print("missing")
            else:
                m.append(st.mean(df['Heart Rate'][index1:index2]))
                x.append(df['Heart Rate'][index1:index2])
                c+=1
        else:
            df1=pd.read_csv(base2+Folder[i]+'/'+startDate[i]+'.csv')
            df2=pd.read_csv(base2+Folder[i]+'/'+endDate[i]+'.csv')
            index1=findIndex(df1['Time'],startTime[i],study)
            index2=findIndex(df2['Time'],endTime[i],study)
            m1=st.mean(df1['Heart Rate'][index1:])
            m2=st.mean(df1['Heart Rate'][:index2])
            l1=len(df1['Heart Rate'][index1:])
            l2=len(df1['Heart Rate'][:index2])
            m3=(m1*l1+m2*l2)/(l1+l2)
            sx=df1['Heart Rate'][index1:]
            sy=df2['Heart Rate'][:index2]
            sz=[]
            sz.append(sx)
            sz.append(sy)
            x.append(np.concatenate(np.array(sz),axis=0).tolist())
            m.append(m3)
            c+=1
    HR= pd.DataFrame({'Subject ID': subjectID, 'Mean HR': m})
#     print(HR)
    return x
    
def appendBetween(x,y,data,j):
    """
    Arguments:
        x (string): initial time pointer from where we need to start appending (H1:M1)
        y (string): last time pointer until when we append (H2:M2)
        data (data frame): from which Heart Rate data needs to be appended
        j (int): represents the subject id
    Output:
        chopped_data (array): new list consisting of Heart Rate data between start and end
    """
    h1,m1=x.split(':')
    h2,m2=y.split(':')
    
    chopped_data=[]
    for i in range(len(data['Time'][j])):
        h,m,s=data['Time'][j][i].split(':')
        if((int(h)>int(h1) or (int(h)==int(h1) and int(m)>int(m1))) and (int(h)<int(h2) or (int(h)==int(h2) and int(m)<int(m2)))):
            chopped_data.append(data['Heart Rate'][j][i])
    
    return chopped_data

def Add(a,b): #used to add a time to b time
    """
    Arguments:
        a (string): represents time 1
        b (string): represents time 2
    Output:
        (string): sum of time 1 + time 2
    """
    h1,m1,s1=a.split(":")
    h2,m2=b.split(":")
    
    h3=int(h1)+int(h2)
    if(int(m1)+int(m2)>60):
        m3=int(m1)+int(m2)-60
        h3=h3+1
    else:
        m3=int(m1)+int(m2)
    
    return str(h3)+":"+str(m3)


    
def fitbitHR(startDate, endDate, startTime, endTime, subjectID, Folder):
    """
     Arguments:
        startDate (string): start date of sleep
        endDate(string): end date of sleep
        startTime(string): start time of sleep
        endTime(string): end time of sleep
        subjectID (int): could be from 0-31
        Folder (string): Folder could be Pixel, Samsung etc. representing the set of devices the subject used
    Output:
        x (list): containing all the data of sleep of one day of that subject
    """
    x=[]
#   subject wise data in x
    time=[]
    for i in range(len(startDate)):
        if(startDate[i]==endDate[i]):
            
            df=pd.read_csv(base2+Folder[i]+'/'+startDate[i]+'.csv')
            index1=findIndex(df['Time'],startTime[i],2)
            index2=findIndex(df['Time'],endTime[i],2)
            
            x.append(df['Heart Rate'][index1:index2].tolist())
            time.append(df['Time'][index1:index2].tolist())

        else:
            df1=pd.read_csv(base2+Folder[i]+'/'+startDate[i]+'.csv')
            df2=pd.read_csv(base2+Folder[i]+'/'+endDate[i]+'.csv')
            index1=findIndex(df1['Time'],startTime[i],2)
            index2=findIndex(df2['Time'],endTime[i],2)
            
            sx=df1['Heart Rate'][index1:]
            sy=df2['Heart Rate'][:index2]
            sz=[]
            sz.append(sx)
            sz.append(sy)
            x.append(np.concatenate(np.array(sz),axis=0).tolist())
            
            tx=df1['Time'][index1:]
            ty=df2['Time'][:index2]
            tz=[]
            tz.append(tx)
            tz.append(ty)
            time.append(np.concatenate(np.array(tz),axis=0).tolist())
        
    newDf=pd.DataFrame({'Time':time,'Heart Rate': x})
#     print(newDf['Time'][3] )
    return newDf

def plot_subject_wise():
    baseFitbit="../../../Data/Physiological_Data/Processed_Data/Fitbit/"
    plots="../../../Plots/Physiological_Plots/Fitbit/HR/"
    
    base2=baseFitbit
    # study 1 reading data
    df=pd.read_csv(base2+'SleepHR/'+'fitbitHR1.csv')

    xr=fitbitMeanHR(df['Start Date'],df['End Date'], df['Start Time'], df['End Time'], df['Subject '],df['Folder'],1)
    # study 2 reading data
    df=pd.read_csv(base2+'SleepHR/'+'fitbitHR2.csv')

    startTime=[]
    startDate=[]
    endTime=[]
    endDate=[]
    # print(df)
    for i in range(len(df['Start'])):
        sD,sT=df['Start'][i].split()
        sT=re.split(':|P|A',sT)
        H= int(sT[0])+12 if int(sT[0])> 7 and int(sT[0])<12 else int(sT[0])
        startTime.append(str(H)+':'+sT[1])
        startDate.append(sD)
        eD,eT=df['End'][i].split()
        eT=re.split('P|A',eT)
        endTime.append(eT[0])
        endDate.append(eD)

    xr2=fitbitMeanHR(startDate,endDate, startTime, endTime, df['Subject'],df['Folder'],2)

    # now we have data from study 1 (xr) and study 2 (xr2)
    # We calculate the means and standard error means of the data

    HRMeanControl=[]
    HRMeanScent=[]

    for i in range(0,len(xr),2):
        
        HRMeanControl.append(st.mean(xr[i]))
        HRMeanScent.append(st.mean(xr[i+1]))

    for i in range(0,len(xr2),2):
        
        HRMeanControl.append(st.mean(xr2[i]))
        HRMeanScent.append(st.mean(xr2[i+1]))

    # Now we plotthe means and standard error means obtained from fitbit

    x = ["Control","Scent"]
    y = [HRMeanControl,HRMeanScent]

    yerr=[]
    yerr.append(stats.sem(HRMeanControl))
    yerr.append(stats.sem(HRMeanScent))

    y_e=[]
    y_e.append(st.mean(HRMeanControl))
    y_e.append(st.mean(HRMeanScent))
    print(st.mean(y[0]),st.mean(y[1]))
    print(stats.sem(HRMeanControl),stats.sem(HRMeanScent))

    fig, ax = plt.subplots()

    x_e=[-0.1,1.2]

    plt.plot(x, y, '-ok')

    for i in range(0,len(xr),2):
        l1=compress(xr[i])
        l2=compress(xr[i+1])
        Statistic,Pval = stats.ttest_rel(l1,l2)
        Pval=Pval/2
    #     print(Pval)
        y, h, col = HRMeanScent[int(i/2)] , 0, 'k'
        star,col=star_col(Statistic,Pval)
        plt.text(1+ 0.05, y+h, star, ha='center', va='center', color=col)

    Statistic,Pval = stats.ttest_rel(HRMeanControl,HRMeanScent)
    Pval=Pval/2

    plt.errorbar(x_e, y_e, yerr, linestyle='None', marker='s',capsize=5)
    ax.set_xlabel('Conditions')
    ax.set_ylabel('HR values')
    # ax.set_title('Line plot with error bars')
    txt1='M (Control)= ' + '%.3f' % stats.sem(HRMeanControl) + " +/-SEM= "+ '%.3f' % stats.sem(HRMeanControl)+" M (Scent)= " + '%.3f' % stats.sem(HRMeanScent)+" +/-SEM= "+ '%.3f' % stats.sem(HRMeanScent)+"\nP= "+'%.3f' % Pval
    fig.text(.125, -0.1, txt1, ha='left')
    s='HR'
    plt.savefig(plots+'Subject_wise_fitbit'+s,dpi=300,bbox_inches='tight')

    plt.show()


def plot_time_wise(sleepDataC,sleepDataS):
    # Plots the mean of 110 minutes chunks of HR fitbit data
    s='HR'

    fig,ax=plt.subplots()

    x=[110, 220, 330, 425]

    newDataC=[]
    newDataS=[]
    y1=[]
    y2=[]
    yerrC=[]
    yerrS=[]

    for i in range(len(sleepDataC)):
        newDataC.append(np.concatenate(sleepDataC[i]).tolist())
        newDataS.append(np.concatenate(sleepDataS[i]).tolist())
        y1.append(np.concatenate(sleepDataC[i]).mean())
        y2.append(np.concatenate(sleepDataS[i]).mean())
        yerrC.append(stats.sem(np.concatenate(sleepDataC[i]).tolist()))
        yerrS.append(stats.sem(np.concatenate(sleepDataS[i]).tolist()))

    for i in range(len(newDataC)):
        Statistics,Pval=stats.ttest_rel(compress(np.asarray(newDataC[i])),compress(np.asarray(newDataS[i])))
        print(Statistics, Pval)
        Pval=Pval/2
        if(Pval<0.05):
            x1, x2 = x[i], x[i]
            y, h, col = max(y1[i],y2[i]), 0.3, 'k'
            if(Statistics<0):
                print("red significant")
                col='r'
            else:
                print("black significant")
                plt.text((x1+x2)*.5, y+h, "*", ha='center', va='center', color=col)
                
    plt.errorbar(x, y1, yerrC, marker='_',capsize=5,color='black',label='control Sleep',barsabove=True)
    plt.errorbar(x, y2, yerrS, marker='_',capsize=5,linestyle='dashed',color='black',label='scent Sleep')
    ax.set_xlabel('Sleep (in minutes)')
    ax.set_ylabel('HR values')
    ax.legend()
    plt.savefig(plots+'Activity_wise_time_axis_fitbit'+s,dpi=300,bbox_inches='tight')
    fig.show()

def make_chunks():
    base2="../../../Data/Physiological_Data/Processed_Data/Fitbit/"
    df=pd.read_csv(base2+'SleepHR/fitbitHR2.csv')

    startTime=[]
    startDate=[]
    endTime=[]
    endDate=[]
    # print(df)
    for i in range(len(df['Start'])):
        sD,sT=df['Start'][i].split()
        sT=re.split(':|P|A',sT)
        H= int(sT[0])+12 if int(sT[0])> 7 and int(sT[0])<12 else int(sT[0])
        startTime.append(str(H)+':'+sT[1])
        startDate.append(sD)
        eD,eT=df['End'][i].split()
        eT=re.split('P|A',eT)
        endTime.append(eT[0])
        endDate.append(eD)


    newDf=fitbitHR(startDate,endDate, startTime, endTime, df['Subject'],df['Folder'])

    intervals=['0:0','1:50','3:40','5:30','7:05']

    sleepDataC=[[],[],[],[]]
    sleepDataS=[[],[],[],[]]
    for j in range(len(newDf)):
    #     print((newDf['Time'][j])[0].split(':'))
        start=newDf['Time'][j][0]
        if j%2==0:
            for i in range(len(intervals)-1):
                x=Add(start,intervals[i])
                y=Add(start,intervals[i+1])
                sleepDataC[i].append(appendBetween(x,y,newDf,j))
        else:
            for i in range(len(intervals)-1):
                x=Add(start,intervals[i])
                y=Add(start,intervals[i+1])
                sleepDataS[i].append(appendBetween(x,y,newDf,j))

    return sleepDataC,sleepDataS
