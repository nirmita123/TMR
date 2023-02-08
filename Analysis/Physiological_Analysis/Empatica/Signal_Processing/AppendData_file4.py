# chop off the noise before appending as timeline is going to fuck up
# append all files
# Things to remember: append from [2:end] timestamps will shift

import pandas as pd
from pandas.io.common import EmptyDataError
import os


base="../../../../Data/Physiological_Data/Processed_Data/Empatica/SleepData/"

n=input("No. of files to append: ")
Dates=[]

sID=input("Subject ID: ")
condition="_scent1"

ibit=''
IBI1=[]
IBI2=[]
BVP=[]
EDA=[]
te=[]
HR=[]
thr=[]
HRV=[]
thrv=[]
Tags=[]

for i in range(int(n)):
    date=input("file "+str(i)+" : ")
    dI=pd.read_csv(base+date+"/IBI.csv", header=None)
    ibit=dI[0][0]
    for j in range(1,len(dI)):
        IBI1.append(dI[0][j])
    for j in range(1,len(dI)):
        IBI2.append(dI[1][j])
    dB=pd.read_csv(base+date+"/BVP.csv", header=None)
    if (i==0):
        BVP.append(dB[0][0])
    for j in range(1,len(dB)):
        BVP.append(dB[0][j])
    dE=pd.read_csv(base+date+"/EDA.csv")
    for j in range(2,len(dE)):
        EDA.append(dE['EDA'][j])
        te.append(dE['Timestamp'][j])
    dHR=pd.read_csv(base+date+"/HR.csv")
    for j in range(2,len(dHR)):
        HR.append(dHR['HR'][j])
        thr.append(dHR['Timestamp'][j])
    dHRV=pd.read_csv(base+date+"/HRV.csv")
    for j in range(2,len(dHRV)):
        HRV.append(dHRV['HRV'][j])
        thrv.append(dHRV['Timestamp'][j])
    try:
        dt= pd.read_csv(base+date+'/tags.csv', header=None)
    except EmptyDataError:
        dt =pd.DataFrame()
    for j in range(len(dt)):
        Tags.append(dt[0][j])
        
dfB=pd.DataFrame(BVP)
dfI=pd.DataFrame({ibit: IBI1, 'IBI': IBI2})
dfE=pd.DataFrame({'Timestamp': te, 'EDA': EDA})
dfHR=pd.DataFrame({'Timestamp': thr, 'HR': HR})
dfHRV=pd.DataFrame({'Timestamp': thrv, 'HRV':HRV})
dft=pd.DataFrame(Tags)

# define the name of the directory to be created
path = base+sID+condition

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# save to csv in a new subject ID wise folder
dfB.to_csv(base+sID+condition+"/BVP.csv", index=False)
dfE.to_csv(base+sID+condition+"/EDA.csv", index=False)
dfHR.to_csv(base+sID+condition+"/HR.csv", index=False)
dfHRV.to_csv(base+sID+condition+"/HRV.csv", index=False)
dft.to_csv(base+sID+condition+"/tags.csv", index=False)

# Awake
# Date=["2_703303_A01321_1","2_703298_A01321_2","3_726993_A00E51","4_703483_A01321","5_704274_A01321","6_727024_A00E51","7_705045_A01321","8_727028_A00E51","9_727030_A00E51","11_705833_A01321","13_706721_A01321","14_736764_A01321","15_727042_A00E51","16_707151_A01321_1","16_707157_A01321_2"]
# Sleep
# Date=["1_697875_A01321","2_698287_A01321_1","2_703135_A01321_2","2_698411_A01321_3","3_698981_A01321","4_699194_A00E51","5_699469_A01321_1","5_699861_A01321_2","6_702300_A00E51","7_701296_A01321_1","7_701297_A01321_2","11_701582_A01321_1","11_701799_A01321_2","13_702209_A01321_1","13_702211_A01321_2","14_703309_A01321_1","14_702431_A01321_2","16_702874_A00E51_1","16_702986_A00E51_2"]

# Date=["2_703317_A01321","3_727025_A00E51","4_704261_A01321","5_704830_A01321_1","5_704912_A01321_2","6_727029_A00E51_1","6_727026_A00E51_2","7_705820_A01321_1","7_705821_A01321_2","9_705821_A00E51_1","9_727033_A00E51_2","11_705921_A01321_1","11_706139_A01321_2","11_706157_A01321_3","13_707098_A01321_1","13_707099_A01321_2","14_736770_A01321","15_727041_A00E51_1","15_727047_A00E51_2","15_727043_A00E51_3"]
# Date=["2_c","2_s","3_c","3_s","4_c","4_s","5_c","5_s","6_c","6_s"]


