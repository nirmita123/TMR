import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from matplotlib import pyplot
from scipy.stats import shapiro
from scipy import stats
import matplotlib.pyplot as plt
import statistics as st
import matplotlib.gridspec as gridspec

base2='../../../Reminessence_Subjective_Analysis/Plots/LanguagePlots/2D_plots/'

#Translations (Responses) in the order R1C1 R1C2 ..RnCn, where Ri is the ith row and Ci is the ith column
translation_scent=[
    "Pinzas",
    "Sierra",
    "Pincel",
    "Hojas",
    "Monedas",
    "Tetera",
    "Rotulador",
    "Peine",
    "Dibujo",
    "Grapadora",
    "Vela",
    "Ventana",
    "Cuerno",
    "Basura",
    "Palillos",
    "Mano",
    "Ascensor",
    "Tierra",
    "Mesa",
    "Gafas"
]
translation_control=[
    "Fuente",
    "Bombilla",
    "Flor",
    "Teclado",
    "Maceta",
    "Taza",
    "Vidrio",
    "Cabeza",
    "Tenedor",
    "Pared",
    "Libro",
    "Cajones",
    "Bolsa",
    "Dulce",
    "Collar",
    "Tijeras",
    "Maquillaje",
    "Calavera",
    "Silla",
    "Bebida"
]



survey=['1. Before sleep + No scent prototype (at the Media Lab)','2. Morning after sleep + No scent prototype (at home)','3. Before sleep + Scent prototype (at the Media Lab)','4. Morning after sleep + Scent prototype (at home)']

def RepresentsInt(s):
    try:
        if int(s)>0:
            return True
    except ValueError:
        return False

def read_results_file(fpath): #reads the results and return an array of results[user][survey][location]
    """
    Arguments:
        fpath (string): path of the .csv results file to be read
    Output:
        results (array): results[user][survey][location] = [w1,t2,w2,t2]
    """
    results=[]
    
    col_list = [52]
    for x in range (112,131):
        col_list.append(x)
   
    df=pd.read_excel(fpath,header=None)
    nline = 1
    for i in range(1,33):
        subject_results = [None]*4
        rows = df[df[2]==i]
        for j in range(2):
            if i>=17:
                row = [str(x.iloc[0]) for no,x in (df[(df[2]==i) & (df[3]==survey[j])].iloc[:,171:191]).iteritems()]
            else:
                row = [str(x.iloc[0]) for no,x in (df[(df[2]==i) & (df[3]==survey[j])].iloc[:,col_list]).iteritems()]
            subject_results[j] = row
        for j in range(2,4):
            if i>=17:
                row = [str(x.iloc[0]) for no,x in (df[(df[2]==i) & (df[3]==survey[j])].iloc[:,col_list]).iteritems()]
            else:
                row = [str(x.iloc[0]) for no,x in (df[(df[2]==i) & (df[3]==survey[j])].iloc[:,171:191]).iteritems()]
            subject_results[j] = row
        results.append(subject_results)
    return results
    
score_control_trans=[]
score_scent_trans=[]

def score_user_survey(responses, typ, user_id): # Computes scores for all locations
    """
    Arguments:
        responses (array): all responses for an individual/survey
        typ (string): either "control" or "scent"
        user_id (int): the user number in order(0-31)
    Output:
        score_condition_trans (int): the calculated score of each user
    """
    #Gets a row (user, survey) and returns average levenstien score
    score_condition_trans = 0

    for i,resp in enumerate(responses): # resp = [word1, trans1, word1,trans2]
        
        if user_id < 16:
            truth = translation_scent[i] if typ == 'scent' else translation_control[i]
        else:
            truth = translation_control[i] if typ == 'scent' else translation_scent[i]
        
        score_trans = 1 if resp == truth else 0
        score_condition_trans  += score_trans
        
        if typ=='control':
            score_control_trans.append(score_trans)
        else:
            score_scent_trans.append(score_trans)

    return score_condition_trans

 
def aggregate_results(all_results):
    """
    Arguments:
        all_results: combined responses of all the 32 users of all 4 surveys (Pre-sleep control, Post-sleep control, Pre-sleep scent, Post-sleep scent)
    Output:
        scores (array size #users*4 X 4): contains [user_id (0-31), survey_id (1-4), typ (control/scent), strans (translation memorization score)]
    """
    scores = []
#    print('{:>8} {:>8} {:>8} {:>8}'.format('User', 'Survey', 'Type', 'Score T'))
    for user_id, results_user in enumerate(all_results):
        for survey_id, results_survey in enumerate(results_user):
            typ = 'control' if survey_id in [0,1] else 'scent'
            strans = score_user_survey(results_survey, typ, user_id)
#            print('{:>8} {:>8} {:>8} {:>8}'.format(user_id, survey_id, typ,strans))
            scores.append([user_id, survey_id, typ, strans])
            
    return scores, score_control_trans, score_scent_trans

