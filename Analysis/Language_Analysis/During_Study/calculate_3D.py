
import pandas as pd
import numpy as np
import seaborn as sns
import distance
import researchpy as rp
import pingouin as pg
from scipy import stats
import math

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

# subjects in the corresponding order
Order_1=[1,2,3,4,5,6,10,15,17,18,19,20,21,22,27,31]
Order_2=[7,8,9,11,12,13,14,16,23,24,25,26,28,29,30,32]

# list of male and female subjects
male = [1,2,3,5,7,8,12,16,19,20,23,24,27,29,31,32]
female = [4,6,9,10,11,13,14,15,17,18,21,22,25,26,28,30]

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


survey=['1. Before sleep + No scent prototype (at the Media Lab)','2. Morning after sleep + No scent prototype (at home)','3. Before sleep + Scent prototype (at the Media Lab)','4. Morning after sleep + Scent prototype (at home)']


def RepresentsInt(s):
    try:
        if int(s)>0:
            return True
    except ValueError:
        return False

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

def read_results_file(fpath): #reads the results and return an array of results[user][survey][location]
    """
    Arguments:
        fpath (string): path of the .csv results file to be read
    Output:
        results (array): results[user][survey][location] = [w1,t2,w2,t2]
    """
    results=[]
    
    df=pd.read_excel(fpath,header=None)
    nline = 1
    for i in range(1,33):
        subject_results = [None]*4
        rows = df[df[2]==i]
        for j in range(4):
            row = [str(x.iloc[0]).lower() for no,x in (df[(df[2]==i) & (df[3]==survey[j])].iloc[:,11:51]).iteritems()]
#             print(row)
            survey_id = df[3]
            chunked = chunks(row, 4)
            subject_results[j] = chunked
        results.append(subject_results)
    return results


#### Structures:

# locationObjects["control" | "scent"][loc_i]
# surveyResponses[user_id][survey_id]
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
    if distance.levenshtein(word,truth0) < distance.levenshtein(word,truth1):
        return len(truth0)
    elif distance.levenshtein(word,truth0) == distance.levenshtein(word,truth1):
        return len(truth0)
    else:
        return len(truth1)
    
def compute_score(responses, ground_truth, truth, typ, user): #function: takes as argumemnt one location
    """
    Argumemnts:
        responses (tuple, size 4): the responses of the user (word1,trans1,word2,trans2)
        ground_truth (tuple, size 2): the two objects in the location
        truth (list): the objects in the location along with its synonyms
        typ (string): condition of which we are testing (control/scent)
        user (int): the user id of the subject
    Output:
        returns score_location_object (int), score_location_trans(int)
        The calculated score (for object and translation) corresponding to the loction (for the two objects)
    """
    
    w1,t1,w2,t2 = responses
    threshold_object = 0.5 # the threshold for normalized levenshtien distance for object memorization
    threshold_trans = 0.5 # the threshold for normalized levenshtien distance for translation memorization
    
    ## score for object memorisation (+1 if in the correct location and distance < than levenshtien threshold)
    ## score for translation memorisation (+1 if in the correct translation of the object irrespective of the location and distance < than levenshtien threshold)
    score_location_object = 0
    score_location_trans = 0
    
    
    for word,trans in [[w1,t1],[w2,t2]]:
        
        levenshtein_distance_normalized_object = min(distance.levenshtein(word,truth[0]),distance.levenshtein(word,truth[1]))/max(len(word),get_min(word,truth[0],truth[1]))
        if trans =="nan":
            levenshtein_distance_normalized_trans = 1
        else:
            levenshtein_distance_normalized_trans = min(distance.levenshtein(trans,translations[truth[0]]),distance.levenshtein(trans,translations[truth[1]]))/max(len(trans),get_min(trans,translations[truth[0]],translations[truth[1]]))
        
        if levenshtein_distance_normalized_object <= threshold_object or np.any([t in word for t in truth]):
            # Means word is correctly remembered in this location
            score_location_object += 1
            if levenshtein_distance_normalized_trans <= threshold_trans or np.any([translations[t] in trans for t in ground_truth]):
                # Means translation is correctly remembered in the correct location
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans) #checks if people knew the word
#                print(trans,translations[truth[0]],translations[truth[1]])
        elif levenshtein_distance_normalized_trans <= threshold_trans:
                # Means user doesn't remeber the object but correctly remembers the translation in the correct location
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans)
#                print(trans,translations[truth[0]],translations[truth[1]])
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
            
            if levenshtein_distance_normalized_trans <= threshold_trans or np.any([translations[t] in trans for t in [final_word]]):
                score_location_trans  += 1 #levenstein(trans,translations[word])
                score_location_trans += check_words_people_knew(user, trans)
#                print(trans,translations[final_word])
        if score_location_object>=1:
            score_location_object=1
        else:
            score_location_object=0
            
        if score_location_trans>=1:
            score_location_trans=1
        else:
            score_location_trans=0
            
    return score_location_object, score_location_trans
    
def score_user_survey(responses, typ, user): # Computes scores for all locations
    """
    Arguments:
        responses (array): all responses for an individual/survey
        typ (string): either "control" or "scent"
        user (int): user_id of the subject
    Output:
        score_condition_object (int): total score of the user for object memorization
        score_condition_trans (int): total score of the user for translation memorization
    """
    score_condition_object = 0
    score_condition_trans = 0

    for i,resp in enumerate(responses): # resp = [word1, trans1, word1,trans2]
        if user < 16: #locations for first 16 users were opposite of the next 16 users
            truth = location_scent[i] if typ == 'scent' else location_control[i]
            ground_truth = truth_control[i] if typ == 'control' else truth_scent[i]
        else:
            truth = location_control[i] if typ == 'scent' else location_scent[i]
            ground_truth = truth_scent[i] if typ == 'control' else truth_control[i]
            
        score_obj, score_trans = compute_score(resp, ground_truth, truth, typ, user) #Call function
        score_condition_object += score_obj
        score_condition_trans  += score_trans
        
        if typ=='control':
            score_control_object.append(score_obj)
            score_control_trans.append(score_trans)
        else:
            score_scent_object.append(score_obj)
            score_scent_trans.append(score_trans)
    return score_condition_object, score_condition_trans

 
def aggregate_results(all_results):
    """
    Arguments:
        all_results: combined responses of all the 32 users of all 4 surveys (Pre-sleep control, Post-sleep control, Pre-sleep scent, Post-sleep scent)
    Output:
        scores (array size #users*4 X 8): contains [user_id (0-31), gender (M/F), survey_id (1-4), order (O1/O2), sleep (0/1), condition (control/scent), sobj (object memorization score), strans (translation memorization score)]
    """
    scores = []
    print('{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format('User', 'Gender', 'Survey', 'Order', 'Type', 'Condition', 'Score O', 'Score T'))
    # Iterates over each user and calculates the individual score
    for user_id, results_user in enumerate(all_results):
        
        for survey_id, results_survey in enumerate(results_user):
            condition = 'control' if survey_id in [0,1] else 'scent'
            sleep = 0 if survey_id in [0,2] else 1 # 0 if wake 1 if sleeping
            sobj, strans = score_user_survey(results_survey, condition, user_id)
             
            order = 1 if (user_id + 1) in Order_1 else 2
            gender = 'M' if (user_id+1) in male else 'F'
            print('{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(user_id, gender, survey_id, order, sleep, condition, sobj, strans))
            
            scores.append([user_id, gender, survey_id, order, sleep, condition, sobj, strans])

    return scores
    

def get_elaborate_data():
    # This stores all scores for each location for all the users in a data frame
    # Features = ['User', 'Location','Gender', 'Type', 'Order', 'Day', 'Condition', 'Score_O', 'Score_T']

    all_scores = []
    condition = 0

    # stores the control condition data
    for i in range(len(score_control_object)):
        if i%10 == 0:
            condition += 1
        if i%20 == 0:
            condition = 0
        order = 1 if (int(i/20) + 1) in Order_1 else 2
        day = 1 if (int(i/20) + 1) in Order_1 else 2
        gender = 'M' if (int(i/20) + 1) in male else 'F'
        location = i%10
        all_scores.append([int(i/20), location, gender, condition, order, day, 'control', score_control_object[i], score_control_trans[i]])

    # stores the scent condition data
    for i in range(len(score_control_object)):
        if i%10 == 0:
            condition += 1
        if i%20 == 0:
            condition = 0
        order = 1 if (int(i/20) + 1) in Order_1 else 2
        day = 2 if (int(i/20) + 1) in Order_1 else 1
        gender = 'M' if (int(i/20) + 1) in male else 'F'
        location = i%10
        all_scores.append([int(i/20), location, gender, condition, order, day, 'scent', score_scent_object[i], score_scent_trans[i]])

    df2 = pd.DataFrame(all_scores, columns = ['User', 'Location','Gender', 'Type', 'Order', 'Day', 'Condition', 'Score_O', 'Score_T'])
    
    return df2





