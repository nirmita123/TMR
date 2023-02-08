
# class Activity_Wise():
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
from numpy import convolve
# from curlyBrace import curlyBrace
import matplotlib.gridspec as gridspec


def broken_axis(ylimits):
    """
    Arguments:
       ylimts (tuple, size 2): stores the ylimits of the two broken axis
    Output:
        ax (axis): main axis plotting the bars
        ax2 (axis): secondary axis to plot the kink
    """
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[40, 1])
    fig = plt.figure(figsize=[3,6])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    plt.subplots_adjust(hspace=0.05)

    kwargs = dict(color='k', clip_on=False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ylim2 = ax2.get_ylim()
    ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
    ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])

    dx = .03*(xlim[1]-xlim[0])
    dy = .015*(ylim[1]-ylim[0])/ylimratio
    ax.plot((xlim[0]-dx,xlim[0]+dx), (ylimits[0]-0.5+ylim[0]-dy,ylimits[0]+ylim[0]+dy), **kwargs)

    dy = .015*(ylim2[1]-ylim2[0])/ylim2ratio
    ax2.plot((xlim[0]-dx,xlim[0]+dx), (ylimits[0]-20+ylim2[1]-dy,ylimits[0]+20+ylim2[1]+dy), **kwargs)
    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax.axes.get_xaxis().set_visible(False)

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(ylimits)  # outliers only
    ax2.set_ylim(0, ylimits[0])  # most of the data
    
    return ax,ax2
# save means of 30 values to do t-tests
plots="../../../Plots/Physiological_Plots/Empatica/"
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
    
def freq(s): #used to get the frequency of data
    """
    Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        (float): returns the frequncy of the type of signal
    """
    if s=='HRV':
        return 1.6
    elif s=='EDA':
        return 4
    else:
        return 1

def concat(data):#used to concatenate data in one axis
    """
    Arguments:
        data (array): the array that needs to be concatenated into 1 list
    Output:
        (list): returns the concatenated list
    """
    return np.concatenate(np.array(data),axis=0)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
def plot_with_baseline(s):
    """
     Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array): returns chunked data for different tasks
    """
    # Path for wake and sleep data
    baseAwake1="../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
    DateWake1=["2_control","3_control","5_control","16_control","14_control","11_control","7_control","17_control","19_control","20_control","22_control","23_control","24_control","25_control","26_control","27_control","28_control","29_control","30_control","31_control","32_control","2_scent","3_scent","5_scent","16_scent","14_scent","11_scent","7_scent","17_scent","19_scent","20_scent","22_scent","23_scent","24_scent","25_scent","26_scent","27_scent","28_scent","29_scent","30_scent","31_scent","32_scent"]

    base= baseAwake1
    Date= DateWake1

    control_list=[]
    scent_list=[]
    
    control_mean=[]
    scent_mean=[]
    
    f=freq(s) # multiply each variable with its frequency
    data_to_plot= chunked_data_wake(s)
    
    for i in range(int(len(Date)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base+Date[i]+"/"+s+"_main.csv")
        data_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/"+s+"_main.csv")
        tag_control= pd.read_csv(base+Date[i]+"/tags_2.csv")
        tag_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/tags_2.csv")

        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]

        # gets the timestamp when memorization starts
        t_control=tag_control['colA'][0]
        t_scent=tag_scent['colA'][0]

        # finds index of the tag when memorization starts in the data
        t1=0 #timestamp control
        for j in range(len(data_control)):
            if data_control['Timestamp'][j]==t_control :
                t1=j
                break
        t2=0 #timestamp scent
        for k in range(len(data_scent)):
            if data_scent['Timestamp'][k]==t_scent :
                t2=k
                break
        
        baseline1=t1-int(f*90) if t1>=f*90 else 0
        baseline2=t2-int(f*90) if t2>=f*90 else 0
        control_list.append([(x-data_to_plot[0][i])/data_to_plot[0][i]*100 for x in data_control[s][baseline1:t1+int(f*1740)]])
        scent_list.append([(x-data_to_plot[1][i])/data_to_plot[1][i]*100 for x in data_scent[s][baseline2:t2+int(f*1740)]])
    
    control_list=list(zip(*control_list))
    scent_list=list(zip(*scent_list))
#     print(len(control_list),len(scent_list))

    for i in range(min(len(control_list),len(scent_list))):
        control_mean.append(mean(control_list[i]))
        scent_mean.append(mean(scent_list[i]))
    
    control_mean= movingaverage(control_mean,f*100)
    scent_mean= movingaverage(scent_mean,f*100)
    
    # post memorization
    control_list_memory=[]
    scent_list_memory=[]
    data_to_plot = np.array(data_to_plot).T.tolist()
    
    for i in range(len(data_to_plot)):
        sum_control=0
        sum_scent=0
        for j in [2,3,4]:
            sum_control+=data_to_plot[i][2*j]
            sum_scent+=data_to_plot[i][2*j+1]

        control_list_memory.append(sum_control/5)
        scent_list_memory.append(sum_scent/5)

    Statistic,Pval = stats.ttest_rel(control_list_memory,scent_list_memory)

    Pval=Pval/2
    print("t("+str(len(control_list_memory)-1)+")=",Statistic,"p=",Pval)


    x=np.array(range(len(control_mean)))
    fig, ax = plt.subplots()
    x1, x2 = int(700*f), int(1150*f)
    y, h, col = max(max(scent_mean),max(control_mean)) , 2, 'k'
        
    if(Pval<0.0001):
        star='***'
    elif Pval<0.001:
        star='**'
    elif Pval<0.05:
        star='*'
    if Pval <0.05:
        plt.text((x1+x2)*.5, y+h+h/20, star, ha='center', va='center', color=col)
        plt.plot([x1, x1, x2, x2], [y+h/2, y+(h), y+(h), y+h/2], lw=1.5, c=col)

    plt.plot(x,control_mean,color='#C0C0C0', markersize=1,label='Control')
    plt.plot(x,scent_mean,color='black', markersize=1,label='Scent')
    
    plt.xticks([int(60*f),int(350*f),int(700*f),int(950*f),int(1150*f)],['', 'Mem3D', 'Test3D','Mem2D','Test2D'])
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(s+' values (in %)')
    ax.legend()
    plt.savefig(plots+"Wake/"+'Raw_data_baseline'+s,dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_all_points(s):
    """
     Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array): returns chunked data for different tasks
    """
    # Path for wake and sleep data
    baseAwake1="../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
    DateWake1=["2_control","3_control","5_control","16_control","14_control","11_control","7_control","17_control","19_control","20_control","22_control","23_control","24_control","25_control","26_control","27_control","28_control","29_control","30_control","31_control","32_control","2_scent","3_scent","5_scent","16_scent","14_scent","11_scent","7_scent","17_scent","19_scent","20_scent","22_scent","23_scent","24_scent","25_scent","26_scent","27_scent","28_scent","29_scent","30_scent","31_scent","32_scent"]

    base= baseAwake1
    Date= DateWake1

    control_list=[]
    scent_list=[]
    
    control_mean=[]
    scent_mean=[]
    
    f=freq(s) # multiply each variable with its frequency
    fig, ax = plt.subplots()
    
    for i in range(int(len(Date)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base+Date[i]+"/"+s+"_main.csv")
        data_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/"+s+"_main.csv")
        tag_control= pd.read_csv(base+Date[i]+"/tags_2.csv")
        tag_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/tags_2.csv")

        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]

        # gets the timestamp when memorization starts
        t_control=tag_control['colA'][0]
        t_scent=tag_scent['colA'][0]

        # finds index of the tag when memorization starts in the data
        t1=0 #timestamp control
        for j in range(len(data_control)):
            if data_control['Timestamp'][j]==t_control :
                t1=j
                break
        t2=0 #timestamp scent
        for k in range(len(data_scent)):
            if data_scent['Timestamp'][k]==t_scent :
                t2=k
                break
        
        baseline1=t1-int(f*90) if t1>=f*90 else 0
        baseline2=t2-int(f*90) if t2>=f*90 else 0
        control_list.append(data_control[s][baseline1:t1+int(f*1740)])
        scent_list.append(data_scent[s][baseline2:t2+int(f*1740)])
    
    control_list=list(zip(*control_list))
    scent_list=list(zip(*scent_list))
    print(len(control_list),len(scent_list))
    for i in range(min(len(control_list),len(scent_list))):
        control_mean.append(mean(control_list[i]))
        scent_mean.append(mean(scent_list[i]))
    
    control_mean= movingaverage(control_mean,f*100)
    scent_mean= movingaverage(scent_mean,f*100)
    
    # post memorization
    control_list_memory=[]
    scent_list_memory=[]
    data_to_plot= chunked_data_wake(s)
    data_to_plot = np.array(data_to_plot).T.tolist()
    
    for i in range(len(data_to_plot)):
        sum_control=0
        sum_scent=0
        for j in [2,3,4]:
            sum_control+=data_to_plot[i][2*j]
            sum_scent+=data_to_plot[i][2*j+1]

        control_list_memory.append(sum_control/5)
        scent_list_memory.append(sum_scent/5)

    Statistic,Pval = stats.ttest_rel(control_list_memory,scent_list_memory)

    Pval=Pval/2
    print("t("+str(len(control_list_memory)-1)+")=",Statistic,"p=",Pval)
    
    control_list=list(zip(*control_list))
    scent_list=list(zip(*scent_list))
    
    if s=='HRV':
        start_=[900]
        end_=[1200]
        for k in range(len(start_)):
            control_list_memory=[]
            scent_list_memory=[]

            for i in range(len(control_list)):
                control_list_memory.append(mean(control_list[i][int(f*start_[k]):int(f*end_[k])]))
                scent_list_memory.append(mean(scent_list[i][int(f*start_[k]):int(f*end_[k])]))

            Statistic,Pval = stats.ttest_rel(control_list_memory,scent_list_memory)
            ax.axvspan(int(start_[k]*f)-150, int(end_[k]*f)-150, facecolor='#c2ecde', alpha=0.5)

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
#         curlyBrace(fig, ax, p3, p4,bool_auto=True,str_text=star, color='k')
    
    plt.plot(x,control_mean,color='#C0C0C0', markersize=1,label='Control')
    plt.plot(x,scent_mean,color='black', markersize=1,label='Scent')
    
    plt.xticks([int(60*f),int(350*f),int(700*f),int(950*f),int(1150*f)],['1', '10\n[Mem3D]', '20\n[Test3D]', '25\n[Mem2D]', '30\n[Test2D]'])
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(s+'values')
    ax.legend()
    plt.savefig(plots+"Wake/"+'Raw_data_'+s,dpi=300,bbox_inches='tight')
    plt.show()
    
    
                            
def chunked_data_wake(s):#used to make chunks for pre-sleep data according to tasks["Memorization 3D","Recall 3D","Memorization 2D","Recall 2D"]
    """
     Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array): returns chunked data for different tasks
    """
    # Path for wake and sleep data
    baseAwake1="../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"
    DateWake1=["2_control","3_control","5_control","16_control","14_control","11_control","7_control","17_control","19_control","20_control","22_control","23_control","24_control","25_control","26_control","27_control","28_control","29_control","30_control","31_control","32_control","2_scent","3_scent","5_scent","16_scent","14_scent","11_scent","7_scent","17_scent","19_scent","20_scent","22_scent","23_scent","24_scent","25_scent","26_scent","27_scent","28_scent","29_scent","30_scent","31_scent","32_scent"]


    base= baseAwake1
    Date= DateWake1
    # 2 mintutes of data before start of memorization
    Base_control=[]
    Base_scent=[]

    # 10 minutes of 3D memorization data
    Memorization3d_control=[]
    Memorization3d_scent=[]

    # 5 minutes of 3D recall/exam data
    Recall3d_control=[]
    Recall3d_scent=[]

    # 5 minutes of 2D memorization data
    Memorization2d_control=[]
    Memorization2d_scent=[]

    # 4 minutes of 2D recall/exam data
    Recall2d_control=[]
    Recall2d_scent=[]

    f=freq(s) # multiply each variable with its frequency

    for i in range(int(len(Date)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base+Date[i]+"/"+s+"_main.csv")
        data_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/"+s+"_main.csv")
        tag_control= pd.read_csv(base+Date[i]+"/tags_2.csv")
        tag_scent= pd.read_csv(base+Date[i+int(len(Date)/2)]+"/tags_2.csv")

        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]

        # gets the timestamp when memorization starts
        t_control=tag_control['colA'][0]
        t_scent=tag_scent['colA'][0]

        # finds index of the tag when memorization starts in the data
        t1=0 #timestamp control
        for j in range(len(data_control)):
            if data_control['Timestamp'][j]==t_control :
                t1=j
                break
        t2=0 #timestamp scent
        for k in range(len(data_scent)):
            if data_scent['Timestamp'][k]==t_scent :
                t2=k
                break
        
        baseline=t1-int(f*120) if t1>=f*120 else 0
        # splits the data into different list for each condition - control
        Base_control.append(mean(data_control[s][baseline:t1]))
        Memorization3d_control.append(mean(data_control[s][t1:t1+int(f*600)]))
        Recall3d_control.append(mean(data_control[s][t1+int(f*600):t1+int(f*1200)]))
        Memorization2d_control.append(mean(data_control[s][t1+int(f*1200):t1+int(f*1500)]))
        Recall2d_control.append(mean(data_control[s][len(data_control)-int(f*240):]))
        # splits the data into different list for each condition - scent
        Base_scent.append(mean(data_scent[s][t2-int(f*120):t2]))
        Memorization3d_scent.append(mean(data_scent[s][t2:t2+int(f*600)]))
        Recall3d_scent.append(mean(data_scent[s][t2+int(f*600):t2+int(f*1200)]))
        Memorization2d_scent.append(mean(data_scent[s][t2+int(f*1200):t2+int(f*1500)]))
        Recall2d_scent.append(mean(data_scent[s][len(data_scent[s])-int(f*240):]))

    # final data that can be plot for each condition
    data_to_plot = [Base_control, Base_scent, Memorization3d_control,Memorization3d_scent, Recall3d_control, Recall3d_scent, Memorization2d_control, Memorization2d_scent, Recall2d_control, Recall2d_scent]

    return data_to_plot

def activity_wise_wake(data_to_plot,s,divide=False):
    """
     Arguments:
        data_to_plot (array): array of activity-wise data per subject data_to_plot[activity][subject] (activity = 0-9) corresponding to --> [Base_control, Base_scent, Memorization3d_control, Memorization3d_scent, Recall3d_control, Recall3d_scent, Memorization2d_control, Memorization2d_scent, Recall2d_control, Recall2d_scent]
        s (string): could be "HRV","HR" or "EDA
    Output:
        plots a activity wise graph with significance stars
    """
    plots="../../../Plots/Physiological_Plots/Empatica/"
    # wake conditions
    Condition=['', 'Memorization 3D', 'Test 3D','Memorization 2D','Test 2D']
    fig_name=''
    if divide:
        Condition=['','a', 'Memorization 3D','b','Test 3D','c','Memorization 2D','d','Test 2D','e']
        fig_name='divide'
    x = Condition
    y1=[]
    y2=[]
    yerrC=[]
    yerrS=[]
   

    fig, ax = plt.subplots()

    for i in range(0,int(len(data_to_plot)),2):
        control_mean = st.mean(data_to_plot[i])
        scent_mean = st.mean(data_to_plot[i+1])
        control_sem = stats.sem((data_to_plot[i])) # compress splits the data into 30 chunks
        scent_sem = stats.sem((data_to_plot[i+1])) # compress splits the data into 30 chunks
        
        y1.append(control_mean)
        y2.append(scent_mean)
        yerrC.append(control_sem)
        yerrS.append(scent_sem)
        
        Statistic,Pval = stats.ttest_rel((data_to_plot[i]),(data_to_plot[i+1]))
        Pval=Pval/2
        print("t("+str(len(data_to_plot[i])-1)+")=",Statistic,"p=",Pval)
        
        x1, x2 = i/2 -0.25, (i+1)/2-0.25
        y, h, col = max(control_mean+control_sem,scent_mean+scent_sem) , 0.01, 'k'
        
        # add stars in case of significance (p<0.05)
        if(Pval<0.0001):
            star='***'
        elif Pval<0.001:
            star='**'
        elif Pval<0.05:
            star='*'
        if Pval <0.05:
            if s=='HRV':
                if(Statistic>0):
                    print("red significant")
                    col='r'
                else:
                    print("black significant")
            else:
                if(Statistic<0):
                    print("red significant")
                    col='r'
                else:
                    print("black significant")
            plt.text((x1+x2)*.5, y+h+0.5, star, ha='center', va='center', color=col)


    plt.errorbar(x, y1, yerrC, marker='_',capsize=5,color='#C0C0C0',label='control',barsabove=True)
    plt.errorbar(x, y2, yerrS, marker='_',capsize=5,color='black',label='scent')
    if divide:
        plt.xticks(np.arange(10), ('','', 'Memorization 3D','', 'Test 3D','','Memorization 2D','','Test 2D',''))
    ax.set_xlabel('Activities')
    ax.set_ylabel(s+'values')
    ax.legend()
#     txt1='M (Control)= ' + '%.3f' % stats.sem(MeanControl) + " +/-SEM= "+ '%.3f' % stats.sem(MeanControl)+" M (Scent)= " + '%.3f' % stats.sem(MeanScent)+" +/-SEM= "+ '%.3f' % stats.sem(MeanScent)+"\nP= "+'%.3f' % Pval
#     fig.text(.125, -0.1, txt1, ha='left')
    plt.savefig(plots+"Wake/"+'Activity_wise_'+fig_name+s,dpi=300,bbox_inches='tight')


    plt.show()
def activity_wise_wake_baseline(data_to_plot,s):
    """
     Arguments:
        data_to_plot (array): array of activity-wise data per subject data_to_plot[activity][subject] (activity = 0-9) corresponding to --> [Base_control, Base_scent, Memorization3d_control, Memorization3d_scent, Recall3d_control, Recall3d_scent, Memorization2d_control, Memorization2d_scent, Recall2d_control, Recall2d_scent]
        s (string): could be "HRV","HR" or "EDA
    Output:
        plots a activity wise graph with significance stars
    """
    plots="../../../Plots/Physiological_Plots/Empatica/"
    # wake conditions
    Condition=['', 'Memorization 3D', 'Test 3D','Memorization 2D','Test 2D']
    x = Condition
    y1=[]
    y2=[]
    yerrC=[]
    yerrS=[]
   
    data_to_plot_t=np.array(data_to_plot).T.tolist()
    for i in range(len(data_to_plot_t)):
        control=data_to_plot_t[i][0]
        scent=data_to_plot_t[i][1]
        for j in range(int(len(data_to_plot_t[0])/2)):
            data_to_plot_t[i][2*j]=data_to_plot_t[i][2*j]-control
            data_to_plot_t[i][2*j+1]=data_to_plot_t[i][2*j+1]-scent
            
    data_to_plot=np.array(data_to_plot_t).T.tolist()        
    fig, ax = plt.subplots()

    for i in range(0,int(len(data_to_plot)),2):
        control_mean = st.mean(data_to_plot[i])
        scent_mean = st.mean(data_to_plot[i+1])
        control_sem = stats.sem((data_to_plot[i])) # compress splits the data into 30 chunks
        scent_sem = stats.sem((data_to_plot[i+1])) # compress splits the data into 30 chunks
        
        y1.append(control_mean)
        y2.append(scent_mean)
        yerrC.append(control_sem)
        yerrS.append(scent_sem)
        
        Statistic,Pval = stats.ttest_rel((data_to_plot[i]),(data_to_plot[i+1]))
        Pval=Pval/2
        print("t("+str(len(data_to_plot[i])-1)+")=",Statistic,"p=",Pval)
        
        x1, x2 = i/2 -0.25, (i+1)/2-0.25
        y, h, col = max(control_mean+control_sem,scent_mean+scent_sem) , 0.01, 'k'
        
        # add stars in case of significance (p<0.05)
        if(Pval<0.0001):
            star='***'
        elif Pval<0.001:
            star='**'
        elif Pval<0.05:
            star='*'
        if Pval <0.05:
            if s=='HRV':
                if(Statistic>0):
                    print("red significant")
                    col='r'
                else:
                    print("black significant")
            else:
                if(Statistic<0):
                    print("red significant")
                    col='r'
                else:
                    print("black significant")
            plt.text((x1+x2)*.5, y+h, star, ha='center', va='center', color=col)


    plt.errorbar(x, y1, yerrC, marker='_',capsize=5,color='#C0C0C0',label='control',barsabove=True)
    plt.errorbar(x, y2, yerrS, marker='_',capsize=5,color='black',label='scent')
    ax.set_xlabel('Activities')
    ax.set_ylabel(s+'values')
    ax.legend()
#     txt1='M (Control)= ' + '%.3f' % stats.sem(MeanControl) + " +/-SEM= "+ '%.3f' % stats.sem(MeanControl)+" M (Scent)= " + '%.3f' % stats.sem(MeanScent)+" +/-SEM= "+ '%.3f' % stats.sem(MeanScent)+"\nP= "+'%.3f' % Pval
#     fig.text(.125, -0.1, txt1, ha='left')
    plt.savefig(plots+"Wake/"+'Activity_wise_baseline'+s,dpi=300,bbox_inches='tight')

    plt.show()




def resting_hr(data,i):
    """
     Arguments:
        data (list): HR data of a subject in a particulation condition (control/scent)
        i (int): subject id
    Output:
        returns a list of change in HR w.r.t resting heart rate
    """
    RestingHR1=[60,62,63,57,61,68,62,65,63,60,62,63,57,61,68,62,65,63]

    return [x-RestingHR1[i] for x in (data)]


def chunked_rest_data_wake(s):
    """
    Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array) : contains activity wise HR data (such as memorization 3D control, recall 3D scent) w.r.t resting HR (each data point is the change  in HR wrt resting HR)
    """
    Date1=["2_control","3_control","4_control","5_control","17_control","20_control","25_control","28_control","30_control","2_scent","3_scent","4_scent","5_scent","17_scent","20_scent","25_scent","28_scent","30_scent"]
    # Path for wake and sleep data
    baseAwake1="../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"

    base1=baseAwake1
    RestingHR1=[60,62,63,57,61,68,62,65,63,60,62,63,57,61,68,62,65,63]


    RestingHR=RestingHR1
    # 2 mintutes of data before start of memorization
    Base_control=[]
    Base_scent=[]

    # 10 minutes of 3D memorization data
    Memorization3d_control=[]
    Memorization3d_scent=[]

    # 5 minutes of 3D recall/exam data
    Recall3d_control=[]
    Recall3d_scent=[]

    # 5 minutes of 2D memorization data
    Memorization2d_control=[]
    Memorization2d_scent=[]

    # 4 minutes of 2D recall/exam data
    Recall2d_control=[]
    Recall2d_scent=[]
    
    f=freq(s) # multiply each variable with its frequency
    
    for i in range(int(len(Date1)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base1+Date1[i]+"/"+s+"_main.csv")
        data_scent= pd.read_csv(base1+Date1[i+int(len(Date1)/2)]+"/"+s+"_main.csv")
        tag_control= pd.read_csv(base1+Date1[i]+"/tags_2.csv")
        tag_scent= pd.read_csv(base1+Date1[i+int(len(Date1)/2)]+"/tags_2.csv")
        
        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]
        
        # gets the timestamp when memorization starts
        t_control=tag_control['colA'][0]
        t_scent=tag_scent['colA'][0]
        
        
       
        # finds index of the tag when memorization starts in the data
        t1=0 #timestamp control
        for j in range(len(data_control)):
            if data_control['Timestamp'][j]==t_control :
                t1=j
                break
        t2=0 #timestamp scent
        for k in range(len(data_scent)):
            if data_scent['Timestamp'][k]==t_scent :
                t2=k
                break
        # splits the data into different list for each condition - control
        Base_control.append(mean(resting_hr(data_control[s][t1-int(f*120):t1],i)))
        Memorization3d_control.append(mean(resting_hr(data_control[s][t1:t1+int(f*600)],i)))
        Recall3d_control.append(mean(resting_hr(data_control[s][t1+int(f*600):t1+int(f*1200)],i)))
        Memorization2d_control.append(mean(resting_hr(data_control[s][t1+int(f*1200):t1+int(f*1500)],i)))
        Recall2d_control.append(mean(resting_hr(data_control[s][len(data_control)-int(f*240):],i)))
        # splits the data into different list for each condition - scent
        Base_scent.append(mean(resting_hr(data_scent[s][t2-int(f*120):t2],i)))
        Memorization3d_scent.append(mean(resting_hr(data_scent[s][t2:t2+int(f*600)],i)))
        Recall3d_scent.append(mean(resting_hr(data_scent[s][t2+int(f*600):t2+int(f*1200)],i)))
        Memorization2d_scent.append(mean(resting_hr(data_scent[s][t2+int(f*1200):t2+int(f*1500)],i)))
        Recall2d_scent.append(mean(resting_hr(data_scent[s][len(data_scent[s])-int(f*240):],i)))

    # final data that can be plot for each condition
    data_to_plot = [Base_control, Base_scent, Memorization3d_control,Memorization3d_scent, Recall3d_control, Recall3d_scent, Memorization2d_control, Memorization2d_scent, Recall2d_control, Recall2d_scent]
                
    return data_to_plot

def chunked_data_divide(s):
    """
    Arguments:
        s (string): could be "HRV", "EDA" or "HR"
    Output:
        data_to_plot (array) : contains activity wise HR data (such as memorization 3D control, recall 3D scent) w.r.t resting HR (each data point is the change  in HR wrt resting HR)
    """
    Date1=["2_control","3_control","5_control","16_control","14_control","11_control","7_control","17_control","19_control","20_control","22_control","23_control","24_control","25_control","26_control","27_control","28_control","29_control","30_control","31_control","32_control","2_scent","3_scent","5_scent","16_scent","14_scent","11_scent","7_scent","17_scent","19_scent","20_scent","22_scent","23_scent","24_scent","25_scent","26_scent","27_scent","28_scent","29_scent","30_scent","31_scent","32_scent"]

    # Path for wake and sleep data
    baseAwake1="../../../Data/Physiological_Data/Processed_Data/Empatica/AwakeData/"

    base1=baseAwake1
    RestingHR1=[60,62,63,57,61,68,62,65,63,60,62,63,57,61,68,62,65,63]


    RestingHR=RestingHR1
    
    chunks_mean = [[] for _ in range(20)]
    
    f=freq(s) # multiply each variable with its frequency
    
    intervals=[-90,-45,0,300,600,900,1200,1300,1320]
    intervals_end=[-240,-120,0]
    for i in range(int(len(Date1)/2)):
        # read the data and tags/timestamps
        data_control= pd.read_csv(base1+Date1[i]+"/"+s+"_main.csv")
        data_scent= pd.read_csv(base1+Date1[i+int(len(Date1)/2)]+"/"+s+"_main.csv")
        tag_control= pd.read_csv(base1+Date1[i]+"/tags_2.csv")
        tag_scent= pd.read_csv(base1+Date1[i+int(len(Date1)/2)]+"/tags_2.csv")
        
        data_control = data_control[np.isfinite(data_control[s])]
        data_scent = data_scent[np.isfinite(data_scent[s])]
        
        # gets the timestamp when memorization starts
        t_control=tag_control['colA'][0]
        t_scent=tag_scent['colA'][0]
        
        # finds index of the tag when memorization starts in the data
        t1=0 #timestamp control
        for j in range(len(data_control)):
            if data_control['Timestamp'][j]==t_control :
                t1=j
                break
        t2=0 #timestamp scent
        for k in range(len(data_scent)):
            if data_scent['Timestamp'][k]==t_scent :
                t2=k
                break
        for j in range(len(intervals)-1):
            start1=0 if(t1+int(f*intervals[j])<0) else t1+int(f*intervals[j])
            start2=0 if(t2+int(f*intervals[j])<0) else t2+int(f*intervals[j])
            chunks_mean[2*j].append(mean(data_control[s][start1:t1+int(f*intervals[j+1])]))
            chunks_mean[2*j+1].append(mean(data_scent[s][start2:t2+int(f*intervals[j+1])]))

        for j in range(8,10,1):
            
            chunks_mean[2*j].append(mean(data_control[s][len(data_control)+int(f*intervals_end[j-8]):len(data_control)+int(f*intervals_end[j-7])]))
            chunks_mean[2*j+1].append(mean(data_scent[s][len(data_scent)+int(f*intervals_end[j-8]):len(data_scent)+int(f*intervals_end[j-7])]))
        
               
    return chunks_mean