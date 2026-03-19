


import pandas as pd
from word2number import w2n
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





def read_split_data(filename):
    file = open(filename, "r").read()
    list_rows = []
    file = file.split("\n")
    for row in range(0, len(file) - 1):
        dict_rows = {}
        splits = file[row].split(" ")
        for s in range(len(splits)):
            if s == 0:
                if int(splits[s]) == 1:
                    dict_rows[s] =  int(splits[0])
                else:
                    dict_rows[s] = -1
            else:
                index,val = [float(e) for e in splits[s].split(':')]
                dict_rows[index] = val
        list_rows.append(dict_rows)
    df =  pd.DataFrame.from_dict(list_rows)
    df = df.fillna(0)
    return df




df_glove_train = read_split_data('glove.train.libsvm')
df_glove_test = read_split_data('glove.test.libsvm')





dataset =  pd.read_csv('misc-attributes-train.csv')
eval =  pd.read_csv('misc-attributes-eval.csv')
test = pd.read_csv('misc-attributes-test.csv')





defendants_age = []
defendant_gender=[]
num_victims=[]
offence_category=[]
offence_subcategory=[]





def change_age(x):
    
    for i,age in enumerate(x['defendant_age']):
#         print("age",age)
        # print(i)
        age = age.strip("(  ) -")
        if age != "not known":
            age = age.replace("years","")
            age = age.replace("about", "")
            age = age.replace("age","")
            age = age.replace("of", "" )
            age = age.replace("old", "")
            age = age.strip()
            if age.find(" ") >= 0:
                temp = age.split(" ")
                # print(temp)
                age = '-'.join(temp)
            syns = wordnet.synsets(age.strip())
            # print("A",age)
            # print("S",syns[0].lemmas()[0].name())
            age = syns[0].lemmas()[0].name()
            if age.find("-") >= 0:
                temp = age.split("-")
                # print(temp)
                age = ' '.join(temp)
            # print(age.strip())
            defendants_age.append(w2n.word_to_num(age.strip()))
        else:
            defendants_age.append(int(0))





change_age(dataset)





gender = ['female', 'indeterminate', 'male']
off_cat = ['breakingPeace','damage','deception','kill','miscellaneous','royalOffences','sexual','theft','violentTheft']
off_sub_cat = ['animalTheft','arson','assault','assaultWithIntent','assaultWithSodomiticalIntent','bankrupcy','bigamy','burglary','coiningOffences',
 'concealingABirth', 'conspiracy', 'embezzlement','extortion','forgery','fraud','gameLawOffence','grandLarceny','highwayRobbery',
 'housebreaking','illegalAbortion','indecentAssault','infanticide','keepingABrothel','kidnapping','libel','mail','manslaughter',
 'murder','other','perjury','pervertingJustice','pettyLarceny','pettyTreason','piracy','pocketpicking','rape','receiving',
 'religiousOffences','returnFromTransportation','riot','robbery','seditiousLibel','seditiousWords','seducingAllegiance',
 'shoplifting','simpleLarceny','sodomy','stealingFromMaster','taxOffences','theftFromPlace','threateningBehaviour','treason',
 'wounding','coiningOffences','vagabond']





def factorize_gender(x):
    for i,g in enumerate(x['defendant_gender']):
        if g in gender:
            defendant_gender.append(gender.index(g))
#     return defendant_gender
            





def factorize_offence_category(x):
    for i,off in enumerate(x['offence_category']):
        if off in off_cat:
            offence_category.append(off_cat.index(off))
#     return offence_category

            
    





def factorize_offence_subcategory(x):
    for i,off_sub in enumerate(x['offence_subcategory']):
        if off_sub in off_sub_cat:
            offence_subcategory.append(off_sub_cat.index(off_sub))
#     return offence_subcategory
            





num_victims = list(dataset['num_victims'])





factorize_gender(dataset)
factorize_offence_category(dataset)
factorize_offence_subcategory(dataset)





df = pd.DataFrame(list(zip(defendants_age,defendant_gender,num_victims,offence_category,offence_subcategory)),  columns =["defendants_age" ,"defendants_gender","num_victims","offence_category","offence_sub_category"])





df





train_numpy = df.iloc[:,:].values





train_label = df_glove_train.iloc[:,0].values





sc = StandardScaler()





X_train = sc.fit_transform(train_numpy)





regressor = RandomForestClassifier(n_estimators=300,random_state=0,max_depth=7)





regressor.fit(X_train,train_label)





y_pred = regressor.predict(X_train)
accuracy_score(train_label,y_pred)





##### for testing ######
defendants_age = []
defendant_gender=[]
num_victims=[]
offence_category=[]
offence_subcategory=[]





change_age(test)
factorize_gender(test)
factorize_offence_category(test)
factorize_offence_subcategory(test)





num_victims = list(test['num_victims'])





df_test = pd.DataFrame(list(zip(defendants_age,defendant_gender,num_victims,offence_category,offence_subcategory)),  columns =["defendants_age" ,"defendants_gender","num_victims","offence_category","offence_sub_category"])





test_numpy = df_test.iloc[:,:].values





test_label = df_glove_test.iloc[:,0].values





X_test = sc.fit_transform(test_numpy)





y_pred_test = regressor.predict(X_test)
accuracy_score(test_label,y_pred_test)





###### for eval #######
defendants_age = []
defendant_gender=[]
num_victims=[]
offence_category=[]
offence_subcategory=[]






factorize_gender(eval)
factorize_offence_category(eval)
factorize_offence_subcategory(eval)
num_victims = list(eval['num_victims'])





for i,age in enumerate(eval['defendant_age']):
#         print("age",age)
        # print(i)
        age = age.strip("(  ) -")
        if age != "not known":
            age = age.replace("years","")
            age = age.replace("about", "")
            age = age.replace("age","")
            age = age.replace("of", "" )
            age = age.replace("old", "")
            age = age.replace("Year", "")
            age = age.replace("his", "")
            age = age.replace("Age", "")
            age = age.strip()
            age = age.strip("d")
            
            if age.find(" ") >= 0:
                temp = age.split(" ")
                if "and" in temp or "or" in temp:
                    age = temp[0]
#                     temp.remove("and")
#                 if "months" in temp or "month" in temp:
#                     temp.remove("months")
# #                     temp.remove("month")
                # print(temp)
                else:
                    age = '-'.join(temp)
#             print("**",age)
            syns = wordnet.synsets(age.strip())
#             print("syns",syns)
            # print("A",age)
            # print("S",syns[0].lemmas()[0].name())
            age = syns[0].lemmas()[0].name()
            if age.find("-") >= 0:
                temp = age.split("-")
                # print(temp)
                age = ' '.join(temp)
#             print(age)
#             defendants_age.append(w2n.word_to_num(age.strip()))
            try:
                defendants_age.append(w2n.word_to_num(age.strip()))
            except:
                defendants_age.append(int(0))
        else:
            defendants_age.append(int(0))





df_eval = pd.DataFrame(list(zip(defendants_age,defendant_gender,num_victims,offence_category,offence_subcategory)),  columns =["defendants_age" ,"defendants_gender","num_victims","offence_category","offence_sub_category"])





eval_numpy = df_eval.iloc[:,:].values





eval_label = eval.iloc[:,0].values





X_eval = sc.fit_transform(eval_numpy)





y_pred_eval = regressor.predict(X_eval)





len(offence_subcategory)





import csv





with open('random_forest_final.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(y_pred_eval)):
        if y_pred_eval[i]== -1:
            val = 0
        else:
            val = 1
        writer.writerow([i, val])













