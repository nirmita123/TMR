import pandas as pd
import numpy as np
import seaborn as sns
import distance
from scipy.stats import norm
from matplotlib import pyplot
from scipy.stats import shapiro
from scipy import stats
import matplotlib.pyplot as plt
import statistics as st
import matplotlib.gridspec as gridspec
from numpy.random import seed
from numpy.random import randn

#dictionary
translations = {
"keyboard":"teclado",
"mug":"taza",
"head":"cabeza",
"book":"libro",
"bulb":"bombilla",
"chair":"silla",
"scissors":"tijeras",
"pot":"maceta",
"wall":"pared",
"sweet":"dulce",
"flower":"flor",
"bag":"bolsa",
"fountain":"fuente",
"fork":"tenedor",
"skull":"calavera",
"necklace":"collar",
"makeup":"maquillaje",
"glass":"vidrio",
"drink":"bebida",
"drawers":"cajones",
"glasses":"gafas",
"stapler":"grapadora",
"horn":"cuerno",
"comb":"peine",
"brush":"pincel",
"coins":"monedas",
"soil":"tierra",
"candle":"vela",
"sheets":"hojas",
"table":"mesa",
"hand":"mano",
"teapot":"tetera",
"saw":"sierra",
"elevator":"ascensor",
"chopsticks":"palillos",
"drawing":"dibujo",
"marker":"rotulador",
"tweezers":"pinzas",
"trash":"basura",
"window":"ventana"
}

base2='../../../Reminessence_Subjective_Analysis/Plots/LanguagePlots/'

#list for objects in the control condition, along with their synonyms
location_control = [
    ["keyboard", "mug","cup"],
    ["book","head"],
    ["chair","bulb"],
    ["pot","scissors"],
    ["wall","sweet","candy","chocolate","dulce"],
    ["flower","bag"],
    ["fountain","fork"],
    ["skull","necklace","neckless","scalp","skeleton","brain"],
    ["makeup","glass","eyeshadow"],
    ["drink","drawers","shelf"]
]

#list for objects in the scent condition, along with their synonyms
location_scent = [
    ["glasses", "stapler","glass"],
    ["horn", "comb", "brush"],
    ["brush", "coins", "money"],
    ["soil", "candle", "candel","candoll","plant"],
    ["sheets", "table", "paper"],
    ["hand", "teapot", "teacup"],
    ["saw", "elevator", "escalator"],
    ["chopsticks", "drawing", "painting","board","draw"],
    ["marker", "tweezers", "twittzer","pen","pencil","color pen"],
    ["trash", "window", "bin","garbage"]
]

#list for objects in the control condition - ground truth (without synonyms)
truth_control = [
    ["keyboard", "mug"],
    ["book","head"],
    ["chair","bulb"],
    ["pot","scissors"],
    ["wall","sweet"],
    ["flower","bag"],
    ["fountain","fork"],
    ["skull","necklace"],
    ["makeup","glass"],
    ["drink","drawers"]
]

#list for objects in the scent condition - ground truth (without synonyms)
truth_scent = [
    ["glasses", "stapler"],
    ["horn", "comb"],
    ["brush", "coins"],
    ["soil", "candle"],
    ["sheets", "table"],
    ["hand", "teapot"],
    ["saw", "elevator"],
    ["chopsticks", "drawing"],
    ["marker", "tweezers"],
    ["trash", "window"]
]

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
    with open(fpath, 'r') as f:
        header = f.readline()
        nline = 1
        subject_results=[None]*2
        survey_id= [0,1]*16
        for line in f:
            line = [w.lower() for w in line.strip().split(',')]
            chunked_3D_scent = chunks(line[3:43], 4)
            chunked_3D_control = chunks(line[43:83], 4)
           
            subject_results[0] = chunked_3D_scent
            subject_results[1] = chunked_3D_control
            
            results.append(subject_results)
            subject_results = [None]*2
            
    
    return results

score_control_object=[]
score_scent_object=[]
score_control_trans=[]
score_scent_trans=[]

def check_words_people_knew(user, trans):
#     There are some words that the subjects knew from before hand - so this functions
#     identifies that and doesn't assign point for the same
    """
    user (int): this is the user id of the subject whose responses are being evaluated
    trans (string): the translation
    Output:
        (int) returns -1 if the subject knew the word from before otherwise 0
    """
    score = 0
    if user == 1 and (trans == "maquijaz" or trans == "libre" or trans == "colar"):
            score = -1
    elif user == 14 and (trans == "libro"):
            score = -1
    elif user == 7 and (trans == "basura"):
            score = -1
    elif user == 8 and (trans == "cabeza" or trans == "mano"):
            score = -1
    elif user == 12 and (trans == "mano" or trans == "monedas"):
            score = -1
    elif user == 13 and (trans == "cabeza" or trans == "libro" or trans == "teclado" or trans == "fuente" or trans == "bebida" or trans == "mano" or trans == "tierra" or trans == "vela"):
            score = -1
    elif (user == 18 or user == 25) and (trans == "mesa"):
            score = -1
    elif user == 19 and (trans == "libro"):
            score = -1
#    elif user == 25 and trans == "dulce":
#            score = -1
    elif user == 28 and (trans == "mesa" or trans == "mano" or trans == "tierra" or trans == "basura" or trans == "monedas" or trans == "taza" or trans == "cabeza" or trans == "flor" or trans == "bebida" or trans == "libro" or trans == "dulce"):
            score = -1
    return score

def get_min(word,truth0,truth1):
#     This returns the length of the word with minimum leveneshtien distance from the response
    """
    word (string): the response from user correspoinding to a location
    truth0 (string): object 1 in the location
    truth1 (string): object 2 in the location
    """
    if distance.levenshtein(word,truth0)<distance.levenshtein(word,truth1):
        return len(truth0)
    else:
        return len(truth1)

def compute_score(responses, ground_truth, truth, typ, user): #function: takes as argumemnt one location
    """
    Argumemnts:
        responses (tuple, size 4): the responses of the user (word1,trans1,word2,trans2)
        truth (tuple, size 2): the two objects in the location
    Output:
        returns score_location_object (int), score_location_trans(int)
        The calculated score (for object and translation) corresponding to the loction (for the two objects)
    """
    
    w1,t1,w2,t2 = responses
    
    threshold=0.5
    ## First score whether object (in English) is in correct location
    score_location_object = 0
    score_location_trans = 0
    for word,trans in [[w1,t1],[w2,t2]]:

        levenshtein_distance_normalized_object = min(distance.levenshtein(word,truth[0]),distance.levenshtein(word,truth[1]))/max(len(word),get_min(word,truth[0],truth[1]))
        if trans =="nan":
            levenshtein_distance_normalized_trans = 1
        else:
            levenshtein_distance_normalized_trans = min(distance.levenshtein(trans,translations[truth[0]]),distance.levenshtein(trans,translations[truth[1]]))/max(len(trans),get_min(trans,translations[truth[0]],translations[truth[1]]))
        
        if levenshtein_distance_normalized_object <= threshold or np.any([t in word for t in truth]):
            # Means word is correctly remembered in this location
            score_location_object += 1
            if levenshtein_distance_normalized_trans <= threshold or np.any([translations[t] in trans for t in ground_truth]):
                # Means translation is correctly remembered in the correct location
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans) #checks if people knew the word
#                 print(trans,translations[truth[0]],translations[truth[1]])
        elif levenshtein_distance_normalized_trans <= threshold:
                # Means user doesn't remeber the object but correctly remembers the translation in the correct location
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans)
#                 print(trans,translations[truth[0]],translations[truth[1]])
        else:
            # Means that the object is wrong and translation is not for the right location
            # Gives the score if translation of the word written is correct
            
            # Find the word with minimum levenshtien distance and then compare its translation
            locations = location_scent if typ == 'scent' else location_control
            low = float('inf')
            final_word = ''
            for i, words in enumerate(locations):
                dist1 = distance.levenshtein(word,words[0])
                dist2 = distance.levenshtein(word,words[1])
                if dist1 < dist2 and dist1 < low:
                    low = dist1
                    final_word = words[0]
                elif dist2 < low:
                    low = dist2
                    final_word=words[1]
                    if trans =="nan":
                        levenshtein_distance_normalized_trans = 1
                    else:
                        levenshtein_distance_normalized_trans = distance.levenshtein(trans,translations[final_word])/max(len(trans),len(translations[final_word]))
            
            if levenshtein_distance_normalized_trans <= threshold or np.any([translations[t] in trans for t in [final_word]]):
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans)
#                 print(user,trans,translations[final_word])
        if score_location_object>=1:
            score_location_object=1
        else:
            score_location_object=0
            
        if score_location_trans>=1:
            score_location_trans=1
        else:
            score_location_trans=0
            
    return score_location_object, score_location_trans
    
def score_user_survey(responses, typ, user_id): # Computes scores for all locations
    """
    Arguments:
        responses (array): all responses for an individual/survey
        typ (string): either "control" or "scent"
        user (int): user_id of the subject
    Output:
        score_condition_object (int): total score of the user for object memorization
        score_condition_trans (int): total score of the user for translation memorization
    """
    #Gets a row (user, survey) and returns average levenstien score
    score_condition_object = 0
    score_condition_trans = 0

    
    for i,resp in enumerate(responses): # resp = [word1, trans1, word1,trans2]
        if user_id < 16: #locations for first 16 users were opposite of the next 16 users
            truth = location_scent[i] if typ == 'scent' else location_control[i]
            ground_truth = truth_control[i] if typ == 'control' else truth_scent[i]
        else:
            truth = location_control[i] if typ == 'scent' else location_scent[i]
            ground_truth = truth_scent[i] if typ == 'control' else truth_control[i]

        score_obj, score_trans = compute_score(resp, ground_truth, truth, typ, user_id) #Call function
        score_condition_object += score_obj
        score_condition_trans  += score_trans
        if typ=='control':
            score_control_object.append(score_obj)
            score_control_trans.append(score_trans)
        else:
            score_scent_object.append(score_obj)
            score_scent_trans.append(score_trans)
    # TODO: Decide whether to divide by 20 to get average score.
    return score_condition_object, score_condition_trans

 
def aggregate_results(all_results):
    """
    Arguments:
        all_results: combined responses of all the 32 users of all 2 surveys (control, scent)
    Output:
        scores (array size #users*2 X 3): contains [condition (control/scent), sobj (object memorization score), strans (translation memorization score)]
    """
    scores = []
#    print('{:>8} {:>8} {:>8}'.format('Type', 'Score O', 'Score T'))
    c=0

    for user_id, results_user in enumerate(all_results):
        #scores_user = []

        for survey_id,results_survey in enumerate(results_user):
            typ = 'control' if survey_id in [0] else 'scent'

            sobj, strans = score_user_survey(results_survey, typ, user_id)
            #scores_user.append(sobj)
            #scores_user.append(strans)
            # equivalently: scores_user += [sobj, strans]
#            print('{:>8} {:>8} {:>8}'.format(typ,sobj,strans))
            scores.append([typ,sobj,strans])

        #scores.append(scores_user)
    return score_control_object, score_scent_object, score_control_trans, score_scent_trans, scores
    
