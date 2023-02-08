import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import statistics as st
import matplotlib.gridspec as gridspec
from matplotlib import pyplot
from scipy import stats

base2='../../../Plots/Language_Plots/Week_After/'

# plots significance using paired t-test - for translation memorization (as it satisfies normality)
def plotSignificance(score_control,score_scent,ax):
    """
    Arguments:
       score_control (list): list 1 t-test
       score_scent (list): list 2 t-test
       ax (axis): axis to plot significance bars on
    Output:
        Pval (double): stores the one-tail p-value from t-test
    """
    Statistic, Pval= stats.ttest_rel(score_control,score_scent)

    Pval= Pval/2
    y, h, col = max((score_control.mean())*50+stats.sem(score_control)*50,(score_scent.mean())*50+stats.sem(score_scent)*50), 3, 'k'
    if(Pval<0.0001):
        star="***"
    elif Pval<0.001:
        star="**"
    elif Pval<0.05:
        star="*"
    if Pval<0.05:
        ax.text(0.5 , (y+h), star, ha='center', va='bottom', color=col)
        ax.plot([0.25, 0.25, 0.75, 0.75], [y+h/2, y+(h), y+(h), y+h/2], lw=1.5, c=col)

    return Pval
    
    
def plot_without_broken_axis(title, figure_name, xlabel1, xlabel2, color1,color2, ylabel, data1, data2, ylim):
    # Plots the calculated change and saves the plot
    # title, x-label1, x-label2, y-label
    fig = plt.figure(figsize=[3,6])
    ax = fig.add_subplot()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    barWidth=0.25
#     print("Control",st.mean(change_control),"+/-",np.std(change_control),"Scent",st.mean(change_scent),"+/-",np.std(change_scent))
    mean_control_object= np.mean(data1)
    mean_scent_object= np.mean(data2)
    sem_control_object= stats.sem(data1)
    sem_scent_object= stats.sem(data2)

    # Make the plot
    ax.bar(0.25, mean_control_object, yerr=sem_control_object, color=color1, width=barWidth, edgecolor='black',capsize=5, label=xlabel1,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax.bar(0.75, mean_scent_object, yerr=sem_scent_object, color=color2, width=barWidth, edgecolor='black',capsize=5, label=xlabel2,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    # Add xticks on the middle of the group bars
    ax.set_title(title)

    # ax.set_xlabel('Condition', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    plt.xticks([0.25,0.75], [xlabel1,xlabel2])
    plt.xlim([0,1])
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    # t-test and plot if significant
    P=plotSignificance(np.asarray(data1)/50,np.asarray(data2)/50,ax)

    txt1='M (Control)= ' + '%.3f' % mean_control_object + " +/-SEM= "+ '%.3f' % sem_control_object+"\nM (Scent)= " + '%.3f' % mean_scent_object+" +/-SEM= "+ '%.3f' % sem_scent_object+"\nP= "+ '%.3f' % P
    ax.text(.01, ylim[0]-2.5, txt1, ha='left')
    ax.text(-0.08, ylim[1], "%", ha='left',fontsize = 14)

    plt.savefig(base2+ figure_name ,dpi=300,bbox_inches='tight')
    plt.show()
    

# used to plot the plot with broken axis and only left and bottom axis visible
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


def plot_with_broken_axis(title, figure_name, xlabel1, xlabel2, color1, color2, ylabel, data1, data2, ylim):
    ax,ax2=broken_axis(ylim)
    barWidth=0.25

    # calculate mean and standard error mean
    mean_control_object= st.mean(data1)*50
    mean_scent_object= st.mean(data2)*50
    sem_control_object= stats.sem(data1)*50
    sem_scent_object= stats.sem(data2)*50

    # Make the plot
    ax.bar(0.25, mean_control_object, yerr=sem_control_object, color=color1, width=barWidth, edgecolor='black',capsize=5, label=xlabel1,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax.bar(0.75, mean_scent_object, yerr=sem_scent_object, color=color2, width=barWidth, edgecolor='black',capsize=5, label=xlabel2,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    ax2.bar(0.25, mean_control_object, yerr=sem_control_object, color=color1, width=barWidth, edgecolor='black',capsize=5, label=xlabel1,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax2.bar(0.75, mean_scent_object, yerr=sem_scent_object, color=color2, width=barWidth, edgecolor='black',capsize=5, label=xlabel2,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    # Add xticks on the middle of the group bars
    ax.set_title(title)
    # ax2.set_xlabel('Condition', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    plt.xticks([0.25,0.75], ['Control','Scent'])
    plt.xlim([0,1])
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    ax.text(-0.1, ylim[1]+1, "%", ha='left',fontsize = 14)


    # t-test and plot if significant
    P=plotSignificance(np.asarray(data1),np.asarray(data2),ax)
    if P>0.0001:
        txt1='M ('+xlabel1+')= ' + '%.3f' % mean_control_object + " +/-SEM= "+ '%.3f' % sem_control_object+"\nM ("+ xlabel2+')= ' + '%.3f' % mean_scent_object+" +/-SEM= "+ '%.3f' % sem_scent_object+"\nP= "+ '%.4f' % P
    else:
        txt1='M ('+xlabel1+')= ' + '%.3f' % mean_control_object + " +/-SEM= "+ '%.3f' % sem_control_object+"\nM "+ xlabel2+')= '+ '%.3f' % mean_scent_object+" +/-SEM= "+ '%.3f' % sem_scent_object+"\nP< 0.0001"


    ax.text(.01, ylim[0]-9, txt1, ha='left')
    plt.savefig(base2+figure_name,dpi=300,bbox_inches='tight')


    plt.show()

