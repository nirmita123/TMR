import pandas as pd
import numpy as np
from scipy import stats
import distance

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

base2='../../Reminessence_Subjective_Analysis/Plots/LanguagePlots/'

def chunks(l, n):#used to make chunks of [word1, translation1, word2, translation2] corresponding to each location
    """
    Arguments:
        l (array): response from each survey
        n (int): no. of elements in each chunk
    Output:
        list of [w1,t2,w2,t2] chunks for all 10 locations
    """
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))

def read_results_file(fpath):#reads the results and return an array of results[user][survey][location]
    """
    Arguments:
        fpath (string): path of the .csv results file to be read
    Output:
        results (array): results[user][survey][location] = [w1,t2,w2,t2]
    """
    
    results = []
    
    df=pd.read_csv(fpath,header=None)
    nline = 1
    subject_results = [None]*2
    
    for index, row in df.iterrows():
        if index >30:
            break
        if index > 0:
            chunked_3D_scent = list(row[83:103])
            chunked_3D_control = list(row[103:123])
           
            subject_results[0] = chunked_3D_scent
            subject_results[1] = chunked_3D_control
            
            results.append(subject_results)
            subject_results = [None]*2
                
            
            nline +=1
    
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
            truth = translation_scent[i] if typ == 'control' else translation_control[i]
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
            typ = 'control' if survey_id in [1] else 'scent'
            strans = score_user_survey(results_survey, typ, user_id)
#            print('{:>8} {:>8} {:>8} {:>8}'.format(user_id, survey_id, typ,strans))
            scores.append([user_id, survey_id, typ, strans])
    return score_control_trans, score_scent_trans, scores
