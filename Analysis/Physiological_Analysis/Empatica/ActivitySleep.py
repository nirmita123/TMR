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
from statistics import mean 
# save means of 30 values to do t-tests

def compress(data):
    """
     Arguments:
        data (list): the list which needs to be compressed
    Output:
        x (list, size = 30): returns 30 data points(means) representing the list
    """
    x=[]
    y=int(len(data)/30)
    for j in range(0,y*30,y):
        x.append((data[j:j+y].mean()))

    return x

def freq(s):
    """
     Arguments:
        s (string): could be "HRV","HR" or "EDA
    Output:
        (float): returns the frequency of the signal
    """
    if s=='HRV':
        return 1.6
    elif s=='EDA':
        return 4
    else:
        return 1
    
def compressSleep(s,r):
    """
     Arguments:
        s (string): could be "HRV","HR" or "EDA
        r (int): could 0 or 1; 1 if need to plot w.r.t resting HR else 0
    Output:
        semC (list): standard error mean for control data
        semS (list): standard error mean for scent data
        T (array): subject wise control (T[2*subject_id]) and scent (T[2*subject_id+1]) data
    """
    baseSleep1="../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"
    DateSleep1=["2c", "3c", "4c", "5c", "6c", "17_control", "18_control", "20_control", "25_control", "28_control", "30_control", "2s", "3s", "4s", "5s", "6s", "17_scent", "18_scent", "20_scent", "25_scent", "28_scent", "30_scent"]


    base1=baseSleep1
    Date1=DateSleep1
    
    RestingHR1=[60,62,63,57,67,61,68,68,62,65,63,60,62,63,57,67,61,68,68,62,65,63]
    semC=[]
    semS=[]
    T=[]
    f=freq(s)
    for i in range(0,int(f*19800),int(f*6600)):
        
        check=[]
        for j in range(1,11):
            d1= pd.read_csv(base1+Date1[j]+"/new"+s+".csv")
            check.append(mean(d1[s][i:i+int(f*6600)]-r*RestingHR1[j]))
            
        T.append(np.array(check))
        semC.append(stats.sem(np.array(check)))
        
        check=[]
        for j in range(12,22):
            d2= pd.read_csv(base1+Date1[j]+"/new"+s+".csv")
            check.append(mean(d2[s][i:i+int(f*6600)]-r*RestingHR1[j]))
            
        T.append(np.array(check))
        semS.append(stats.sem(np.array(check)))
    
    
    check=[]
    for j in range(1,11):
        d1= pd.read_csv(base1+Date1[j]+"/new"+s+".csv")
        check.append(mean(d1[s][i:i+int(f*5750)]-r*RestingHR1[j]))
    
    T.append(np.array(check))
    semC.append(stats.sem(np.array(check)))
    
    check=[]
    for j in range(12,22):
        d2= pd.read_csv(base1+Date1[j]+"/new"+s+".csv")
        check.append(mean(d2[s][i:i+int(f*5750)]-r*RestingHR1[j]))
    
    T.append(np.array(check))
    semS.append(stats.sem(np.array(check)))
    
   
    return semC, semS, T

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def plot_all_points_sleep(s):
    """
     Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array): returns chunked data for different tasks
    """
    # Path for wake and sleep data
    plots="../../../Plots/Physiological_Plots/Empatica/"
    baseSleep1="../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"
    DateSleep1=["2c", "3c", "4c", "5c", "6c", "17_control", "18_control", "20_control", "25_control", "28_control", "30_control", "2s", "3s", "4s", "5s", "6s", "17_scent", "18_scent", "20_scent", "25_scent", "28_scent", "30_scent"]


    base= baseSleep1
    Date= DateSleep1

    control_list=[]
    scent_list=[]
    
    control_mean=[]
    scent_mean=[]
    
    f=freq(s) # multiply each variable with its frequency

    for i in range(int(len(Date)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base+Date[i]+"/new"+s+".csv")
        data_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/new"+s+".csv")
       
        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]
        
        control_list.append(data_control[s][:int(f*25500)])
        scent_list.append(data_scent[s][:int(f*25500)])
    
    control_list=list(zip(*control_list))
    scent_list=list(zip(*scent_list))
    print(len(control_list),len(scent_list))
    for i in range(min(len(control_list),len(scent_list))):
        control_mean.append(mean(control_list[i]))
        scent_mean.append(mean(scent_list[i]))
    
    control_mean= movingaverage(control_mean,f*300)
    scent_mean= movingaverage(scent_mean,f*300)
    fig, ax = plt.subplots()
    
    control_list=list(zip(*control_list))
    scent_list=list(zip(*scent_list))
    
#     # post memorization
    if s=='HRV':
        start_=[400,13700]
        end_=[4000,15000]
        for k in range(len(start_)):
            control_list_memory=[]
            scent_list_memory=[]

            for i in range(len(control_list)):
                control_list_memory.append(mean(control_list[i][int(f*start_[k]):int(f*end_[k])]))
                scent_list_memory.append(mean(scent_list[i][int(f*start_[k]):int(f*end_[k])]))

            Statistic,Pval = stats.ttest_rel(control_list_memory,scent_list_memory)
            ax.axvspan(int(start_[k]*f), int(end_[k]*f), facecolor='#c2ecde', alpha=0.5)

            print("M(Control)=",mean(control_list_memory),"M(Scent)=",mean(scent_list_memory))
            print("Sem(Control)=",stats.sem(control_list_memory),"Sem(Scent)=",stats.sem(scent_list_memory))
            Pval=Pval/2
            print("t("+str(len(control_list_memory)-1)+")=",Statistic,"p=",Pval)

   
    x=np.array(range(len(control_mean)))
    
#     x1, x2 = int(350*f), int(1150*f)
#     y, h, col = max(max(scent_mean),max(control_mean)) , 10, 'k'
#     p3,p4=[x1,y+h/20], [x2, y+(h/20)]
    
#     if(Pval<0.0001):
#         star='***'
#     elif Pval<0.001:
#         star='**'
#     elif Pval<0.05:
#         star='*'
#     if Pval <0.05:
#         plt.text((x1+x2)*.5, y+h+2, star, ha='center', va='center', color=col)
#         plt.plot([x1, x1, x2, x2], [y+h/2, y+(h), y+(h), y+h/2], lw=1.5, c=col)
# #         curlyBrace(fig, ax, p3, p4,bool_auto=True,str_text=star, color='k')
    plt.plot(x,control_mean,color='#C0C0C0', markersize=1,label='Control')
    plt.plot(x,scent_mean,color='black', markersize=1,label='Scent')
    
    plt.xticks([int(5100*f),int(10200*f),int(15300*f),int(20400*f)],["1.5\n[~S.C.1]","3\n[~S.C.2]","4.5\n[~S.C.3]", "6\n[~S.C.4]"])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(s+'values')
#     ax.text(0.2, -0.1, 'Sleep Start',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='blue', fontsize=8,fontweight='bold')
#     ax.text(0.465, -0.1, 'Sleep Mid1',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='blue', fontsize=8,fontweight='bold')
#     ax.text(0.692, -0.1, 'Sleep Mid2',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='blue', fontsize=8,fontweight='bold')
#     ax.text(0.94, -0.1, 'Sleep End',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='blue', fontsize=8,fontweight='bold')
    
    
    ax.legend()
    plt.savefig(plots+"Sleep/"+'Raw_data_'+s,dpi=300,bbox_inches='tight')
    plt.show()
    
def activity_wise_sleep(s,r):
    """
     Arguments:
        s (string): could be "HRV","HR" or "EDA
        r (int): could 0 or 1; 1 if need to plot w.r.t resting HR else 0
    Output:
        plots activity wise sleep data, where activities are 'SleepStart', 'SleepMid1','SleepMid2','SleepEnd'
    """
    plots="../../../Plots/Physiological_Plots/Empatica/"

    x = ['SleepStart', 'SleepMid1','SleepMid2','SleepEnd']
    semC,semS,T=compressSleep(s,r)

    y1 = []
    y2 = []

    yerrC=[]
    yerrS=[]

    y_e=[]
    x_e=[-0.1,1.1]

    fig, ax = plt.subplots()

    for i in range(0,int(len(T)),2):
        control_mean = st.mean(T[i])
        scent_mean = st.mean(T[i+1])
        
        control_sem = semC[int(i/2)]
        scent_sem = semS[int(i/2)]
        
        y1.append(control_mean)
        y2.append(scent_mean)
        yerrC.append(control_sem)
        yerrS.append(scent_sem)

        Statistic,Pval = stats.ttest_rel((T[i]),(T[i+1]))
        Pval=Pval/2
#         print(Pval)
        
        x1, x2 = int(i/2), int(i/2)
        y, h, col = max(control_mean+control_sem, scent_mean+scent_sem) , 0.2, 'k'
        star=''
            
        if(Pval<0.0001):
            star='***'
        elif Pval<0.001:
            star='**'
        elif Pval<0.05:
            star='*'
            
        plt.text((x1+x2)*.5, y+h, star, ha='center', va='center', color=col)

    print(y1)
    plt.errorbar(x, y1, yerrC, marker='_',capsize=5,color='#C0C0C0',label='control',barsabove=True)
    plt.errorbar(x, y2, yerrS, marker='_',capsize=5,color='black',label='scent')
    ax.set_xlabel('Activities')
    ax.set_ylabel(s+' values')
    ax.legend()
    # ax.set_title('Line plot with error bars')
#     txt1='M (Control)= ' + '%.3f' % stats.sem(MeanControl) + " +/-SEM= "+ '%.3f' % stats.sem(MeanControl)+" M (Scent)= " + '%.3f' % stats.sem(MeanScent)+" +/-SEM= "+ '%.3f' % stats.sem(MeanScent)+"\nP= "+'%.3f' % Pval
#     fig.text(.125, -0.1, txt1, ha='left')
    plt.savefig(plots+"Sleep/"+'Acitivity_wise_'+s,dpi=300,bbox_inches='tight')


    plt.show()
    
def chunked_data_divide_sleep(s):
    """
    Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array) : contains activity wise HR data (such as memorization 3D control, recall 3D scent) w.r.t resting HR (each data point is the change  in HR wrt resting HR)
    """
    plots="../../../Plots/Physiological_Plots/Empatica/"
    baseSleep1="../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"
    DateSleep1=[ "3c", "4c", "5c", "6c", "17_control", "18_control",  "25_control", "28_control", "30_control",  "3s", "4s", "5s", "6s", "17_scent", "18_scent",  "25_scent", "28_scent", "30_scent"]
    #"2c","2s","20_control","20_scent",

    base= baseSleep1
    Date= DateSleep1
    
    chunks_mean = [[] for _ in range(16)]
    
    f=freq(s) # multiply each variable with its frequency
    
    intervals=[0,3000,5500,8000,10500,13000,15500,17000,22000]
    
    for i in range(int(len(Date)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base+Date[i]+"/new"+s+".csv")
        data_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/new"+s+".csv")
        
        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])] 
#         print(len(data_control),len(data_scent))
        for j in range(len(intervals)-1):
#             print(j,f*intervals[j],f*intervals[j+1])
            start1=0 if(int(f*intervals[j])<0) else int(f*intervals[j])
            start2=0 if(int(f*intervals[j])<0) else int(f*intervals[j])
            chunks_mean[2*j].append(mean(data_control[s][start1:int(f*intervals[j+1])]))
            chunks_mean[2*j+1].append(mean(data_scent[s][start2:int(f*intervals[j+1])]))        
               
    return chunks_mean

def activity_wise_sleep_divide(T,s):
    """
     Arguments:
        s (string): could be "HRV","HR" or "EDA
        r (int): could 0 or 1; 1 if need to plot w.r.t resting HR else 0
    Output:
        plots activity wise sleep data, where activities are 'SleepStart', 'SleepMid1','SleepMid2','SleepEnd'
    """
    plots="../../../Plots/Physiological_Plots/Empatica/"

    x = ['SleepStart','a', 'SleepMid1','b','SleepMid2','c','SleepEnd','d']
    
    y1 = []
    y2 = []

    yerrC=[]
    yerrS=[]

    y_e=[]
    x_e=[-0.1,1.1]

    fig, ax = plt.subplots()

    for i in range(0,int(len(T)),2):
        control_mean = st.mean(T[i])
        scent_mean = st.mean(T[i+1])
        
        control_sem = stats.sem(T[i])
        scent_sem = stats.sem(T[i+1])
        
        y1.append(control_mean)
        y2.append(scent_mean)
        yerrC.append(control_sem)
        yerrS.append(scent_sem)

        Statistic,Pval = stats.ttest_rel((T[i]),(T[i+1]))
        Pval=Pval/2
#         print(Pval)
        
        x1, x2 = int(i/2), int(i/2)
        y, h, col = max(control_mean+control_sem, scent_mean+scent_sem) , 0.2, 'k'
        star=''
            
        if(Pval<0.0001):
            star='***'
        elif Pval<0.001:
            star='**'
        elif Pval<0.05:
            star='*'
            
        plt.text((x1+x2)*.5, y+h, star, ha='center', va='center', color=col)

    print(y1)
    plt.errorbar(x, y1, yerrC, marker='_',capsize=5,color='#C0C0C0',label='control',barsabove=True)
    plt.errorbar(x, y2, yerrS, marker='_',capsize=5,color='black',label='scent')
    plt.xticks(np.arange(8),[3000,5500,8000,10500,13000,15500,17000,22000])
    
    ax.set_xlabel('Time (in seconds)')
    ax.set_ylabel(s+' values')
    ax.legend()
    # ax.set_title('Line plot with error bars')
#     txt1='M (Control)= ' + '%.3f' % stats.sem(MeanControl) + " +/-SEM= "+ '%.3f' % stats.sem(MeanControl)+" M (Scent)= " + '%.3f' % stats.sem(MeanScent)+" +/-SEM= "+ '%.3f' % stats.sem(MeanScent)+"\nP= "+'%.3f' % Pval
#     fig.text(.125, -0.1, txt1, ha='left')
    plt.savefig(plots+"Sleep/"+'Acitivity_wise_divide'+s,dpi=300,bbox_inches='tight')


    plt.show()
