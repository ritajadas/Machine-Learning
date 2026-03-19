


import pandas as pd
from word2number import w2n
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
import math
import csv





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





labels = list(df_glove_train.iloc[:,0])





factorize_gender(dataset)
factorize_offence_category(dataset)
factorize_offence_subcategory(dataset)





df = pd.DataFrame(list(zip(labels,defendants_age,defendant_gender,num_victims,offence_category,offence_subcategory)),  columns =["label","defendants_age" ,"defendants_gender","num_victims","offence_category","offence_sub_category"])





train_numpy = df.to_numpy()





def batch_perceptron(data,hp,ep):
    # initialize weights and bias term randomly between -0.01 and 0.01
    np.random.seed(7)
    update = 0
    av_w = np.zeros(shape=data.shape[1]-1)
    av_b = 0
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    b = np.random.uniform(-0.01, 0.01)
    dict_epoch_acc = {}
    learning_rate = hp
    for i in range(ep):
        accuracy = 0
        np.random.shuffle(data)
        for r in range(len(data)):
            ground_truth = data[r, 0]
            sample = data[r, 1:]
            if np.dot(np.transpose(w), sample) + b <= 0:
                prediction = -1
            else:
                prediction = 1
            if int(ground_truth) != int(prediction):
                update += 1
                w = w + learning_rate * ground_truth * sample
                b = b + learning_rate * ground_truth
            else:
                accuracy += 1
            av_w = av_w + w
            av_b = av_b + b
        dict_epoch_acc[i] = av_w, av_b, (accuracy / len(data))
    return  dict_epoch_acc,update





def evaluate(we, bi, test_data):
    accuracy = 0
    for r in range(len(test_data)):
        ground_truth = test_data[r,0]
        sample = list(test_data[r,:])
        sample.pop(0)
        prediction = -1 if np.dot(np.transpose(we), sample) + bi <= 0 else 1
        if prediction == ground_truth:
            accuracy = accuracy + 1
    return (accuracy / len(test_data)) * 100





def cal_max(dict):
    acc = 0
    we_training = []
    bias_training = 0
    for key, value in dict.items():
        if acc < value[2]:
            acc = value[2]
            we_training = value[0]
            bias_training = value[1]
    return we_training,bias_training,acc





def crossvalidation(f1,f2,f3,f4,f5):

    best_h = 0
    max_acc = 0
    hyper_paramter = [0.1,1,0.01]
    for h in hyper_paramter:
        acc = 0
        #run for f1 as test:
        frames = [f2, f3, f4, f5]
        train =np.concatenate((f2,f3,f4,f5),axis=0)
        d, u = batch_perceptron(train, h, 10)
        w1,b1,a1 = cal_max(d)
        acc = acc + evaluate(w1,b1,f1)
        #run for f2 as test
        frames = [f1, f3, f4, f5]
        train = np.concatenate((f1,f3,f4,f5),axis=0)
        d, u = batch_perceptron(train, h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f2)
        #run for f3 as test
        frames = [f2, f1, f4, f5]
        train = np.concatenate((f1, f2, f4, f5), axis=0)
        d, u = batch_perceptron(train, h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f3)
        #run for f4 as test
        frames = [f2, f1, f3, f5]
        train = np.concatenate((f1, f3, f2, f5), axis=0)
        d, u = batch_perceptron(train, h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f4)
        #run for f5 as test
        frames = [f2, f1, f4, f3]
        train = np.concatenate((f1, f3, f4, f2), axis=0)
        d, u = batch_perceptron(train, h, 10)
        w1, b1, a1 = cal_max(d)
        acc = acc + evaluate(w1, b1, f5)
        if max_acc < acc/5:
            max_acc = acc/5
            best_h = h
    return best_h





fold1 = train_numpy[5000:7500,:]
fold2 = train_numpy[7500:10000,:]
fold3 = train_numpy[10000:12500,:]
fold4 = train_numpy[12500:15000,:]
fold5 = train_numpy[15000:17500,:]





best_lr = crossvalidation(fold1,fold2,fold3,fold4,fold5)





best_lr





weights= {}
bias = {}





train1 = np.copy(train_numpy)





for i in range(200):
    np.random.shuffle(train1)
    sample = train1[:1750,:]    
    dict_training_per,u = batch_perceptron(sample, best_lr, 10)
    w,b,a = cal_max(dict_training_per)
    weights[i] = w
    bias[i] = b





len(weights)





def prediction(we,b,sample):
#     sample.pop(0)
    prediction = -1 if np.dot(np.transpose(we), sample) + b <= 0 else 1
    return prediction





def new_dataset(weights,bias,data):
    l = []
    for i in range(len(data)):
        row = []
        row.append(data[i,0])
        for key,value in weights.items():
            x = data[i,1:]
#             del prediction_rows[:]
            p=prediction(value,bias[key],x)
#             p = prediction_rows[0]
            row.append(p)
        l.append(row)
    return np.array(l)





def new_dataset_eval(weights,bias,data):
    l = []
    for i in range(len(data)):
        row = []
#         row.append(data[i,0])
        for key,value in weights.items():
            x = data[i,:]
#             del prediction_rows[:]
            p=prediction(value,bias[key],x)
#             p = prediction_rows[0]
            row.append(p)
        l.append(row)
    return np.array(l)





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
label= list(df_glove_test.iloc[:,0])





df_test = pd.DataFrame(list(zip(label,defendants_age,defendant_gender,num_victims,offence_category,offence_subcategory)),  columns =["label","defendants_age" ,"defendants_gender","num_victims","offence_category","offence_sub_category"])





test_numpy = df_test.to_numpy()





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
label = [0 for i in range(len(eval))]




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





eval_numpy = df_eval.to_numpy()





train_tree = new_dataset(weights,bias,train_numpy)
test_tree = new_dataset(weights,bias,test_numpy)





eval_tree = new_dataset_eval(weights,bias,eval_numpy)





###### now apply ID3 on this #####
class treenode:
    def __init__(self, featureName):
        #print("featurename",featureName)
        self.featureName = featureName
        self.children = []

def findmajoritylabel(dataset):
    n = list(dataset[:,0]).count(-1)
    p = list(dataset[:,0]).count(1)
    if n > p:
        return -1
    else:
        return 1

def check_all_label_same(dataset):
    if len(dataset) == list(dataset[:,0]).count(1):
        return 1
    if len(dataset) == list(dataset[:,0]).count(-1):
        return -1

def cal_entropy(dataset,attributeval=None, feature=None):
    nrows = len(dataset)
    if feature == None:
        p = list(dataset[:,0]).count(1)
        n = list(dataset[:,0]).count(-1)
        if p / nrows == 0 or n / nrows == 0:
            # Entropy_S = 0
            return 0
        entropy = - ((p / nrows) * math.log((p / nrows), 2)) - ((n / nrows) * math.log((n / nrows), 2))
        return  entropy
    else:
        len_dataset_feature = dataset[dataset[:,feature] == attributeval]
        if len(len_dataset_feature) == 0:
            return  0
        else:
            p = list(len_dataset_feature[:,0]).count(1)/len(len_dataset_feature)
            n = list(len_dataset_feature[:,0]).count(-1)/len(len_dataset_feature)
            if p == 0 or n == 0:
                return 0
            entropy = -  p* math.log(p,2)  - n*math.log(n,2)
            # print(feature,attributeval,(len(len_dataset_feature)/nrows)*entropy)
            return (len(len_dataset_feature)/nrows)*entropy

def cal_max_gain(dataset, feature_set):
    dataset_entropy =  cal_entropy(dataset)
    # print("entropy",dataset_entropy)
    max_gain = -1
    selected_feature = ''
    for f in feature_set:
        entropy_feature = 0
        for all_vals in set(dataset[:,f]):
            entropy_feature = entropy_feature + cal_entropy(dataset,all_vals,f)
        gain = dataset_entropy - entropy_feature
        # print("gain",gain)
        # print(f,gain)
        if max_gain < gain:
            max_gain = gain
            selected_feature = f
    # print(selected_feature)
    return selected_feature





def ID3(dataset,featureset,current_depth,depth):
    # print("length",dataset)
    if current_depth + 1 > depth:
        return treenode(findmajoritylabel(dataset))
    #check if all labels are the same, if yes return a node with the label
    if check_all_label_same(dataset) == -1:  #guilty
        #print("label 0" )
        return treenode(-1)
    if check_all_label_same(dataset) == 1:  #not guilty
        #print("label 1")
        return treenode(1)
    if len(featureset) == 0:
        # print("empty")
        return treenode(findmajoritylabel(dataset))
    #find feature with max info gain
    new_selected_feature = cal_max_gain(dataset, featureset)
    # print("n",new_selected_feature)
    node = treenode(new_selected_feature)
    children = []
    set_of_possible_values = set(dataset[:,new_selected_feature])
    for v in set_of_possible_values:
        sv =  dataset[dataset[:,new_selected_feature] == v]
        if len(sv) == 0:
            # print("entered empty",new_selected_feature,v)
            return treenode(findmajoritylabel(dataset))
        else:
            new_feature_set = featureset.copy()
            new_feature_set.remove(new_selected_feature)
            # childnode = ID3(sv, new_feature_set)
            childnode = ID3(sv, new_feature_set,current_depth+1,depth)
            child_dict = {"value": v, "child": childnode}
            children.append(child_dict)
    node.children = children
    return node





def prediction_t(row, root):
    if not root.children:
        # print(root.featureName)
        prediction_rows.append(root.featureName)
        return root.featureName

    decision_to_take = row[root.featureName]
    # print(root.featureName)
    # print(decision_to_take)
    index = 0
    for i in range(len(root.children)):
        if root.children[i]['value'] == decision_to_take:
            index = i
    prediction_t(row, root.children[index]['child'])






def accuracyCal(data, root):
    list_p = []
    for i in range(0, len(data)):
        # print(i)
        v = prediction_t(list(data[i,:]), root)
        # prediction_rows.append(v)
        # print(v)
    correct_predictions = 0
    wrong_predictiosn = 0
    # print(len(prediction_rows))
    # print(len(data))
    # for i in range(0, len(data)):
    #     if data.iloc[i][0] == prediction_rows[i]:
    #         correct_predictions += 1
    # print((correct_predictions) / len(data))
    # return prediction_rows
    for i in range(0, len(data)):
        if data[i,0] == prediction_rows[i]:
            correct_predictions += 1
    return (correct_predictions) / len(data)




feature_set = [i for i in range(1,201)]
feature_set_eval = [i for i in range(1,200)]





root= ID3(train_tree,feature_set,0,14)





prediction_rows = []





acc = accuracyCal(train_tree,root)





acc





prediction_rows = []





acc = accuracyCal(test_tree,root)




acc





prediction_rows = []





def cal_tree_depth(node,depth):
    if len(node.children) == 0:
        return depth
    max_depth = 0
    for child_node in node.children:
        child_node_depth = cal_tree_depth(child_node['child'], depth + 1)
        if max_depth < child_node_depth:
            max_depth = child_node_depth
    return max_depth





cal_tree_depth(root,0)





def prediction_t_e(row, root):
    if not root.children:
        # print(root.featureName)
        prediction_rows.append(root.featureName)
        return root.featureName

    decision_to_take = row[root.featureName]
    # print(root.featureName)
    # print(decision_to_take)
    index = 0
    for i in range(len(root.children)):
        if root.children[i]['value'] == decision_to_take:
            index = i
    prediction_t_e(row, root.children[index]['child'])






def accuracyCal_e(data, root):
    list_p = []
    for i in range(0, len(data)):
        # print(i)
        v = prediction_t(list(data[i,:]), root)
        # prediction_rows.append(v)
        # print(v)
    correct_predictions = 0
    wrong_predictiosn = 0
    # print(len(prediction_rows))
    # print(len(data))
    # for i in range(0, len(data)):
    #     if data.iloc[i][0] == prediction_rows[i]:
    #         correct_predictions += 1
    # print((correct_predictions) / len(data))
    return prediction_rows
#     for i in range(0, len(data)):
#         if data[i,0] == prediction_rows[i]:
#             correct_predictions += 1
#     return (correct_predictions) / len(data)





p = accuracyCal_e(eval_tree,root)





p





with open('perceptron_id3_ensemble.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(p)):
        if p[i]== -1:
            val = 0
        else:
            val = 1
        writer.writerow([i, val])

