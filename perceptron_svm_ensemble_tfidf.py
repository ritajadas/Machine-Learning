
import pandas as pd
from word2number import w2n
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
import math
import csv
from random import seed





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





df_train = read_split_data('tfidf.train.libsvm')
df_test = read_split_data('tfidf.test.libsvm')
df_eval = read_split_data('tfidf.eval.anon.libsvm')




cols_test = list(df_test.columns)
for i in range(0,10001):
    if i not in cols_test:
        df_test[i] = [0.0 for i in range(len(df_test))]


cols_eval = list(df_eval.columns)
for i in range(0,10001):
    if i not in cols_eval:
        df_eval[i] = [0.0 for i in range(len(df_eval))]




df_test = df_test.reindex(sorted(df_test.columns), axis = 1)
df_train = df_train.reindex(sorted(df_train.columns), axis = 1)
df_eval =  df_eval.reindex(sorted(df_eval.columns),axis = 1)





train_numpy = df_train.to_numpy()
test_numpy = df_test.to_numpy()
eval_numpy =  df_eval.to_numpy()




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





train_svm = new_dataset(weights,bias,train_numpy)
test_svm = new_dataset(weights,bias,test_numpy)




train_numpy.shape





eval_svm = new_dataset(weights,bias,eval_numpy)





def svm(data,lr,C,ep):
    #initialize weights and bias term

    lr_0 = lr
    loss = 0.0
    dict_epoch_accuracy = {}
    dict_epoch_loss = {}
    prev_loss = float("inf")
    seed(7)
    # print(data.shape[1])
    w = np.random.uniform(-0.01,0.01,size=data.shape[1])
    # for i in range(len(data.columns)):
    #     w.append(uniform(-0.01,0.01))
    # print(len(w))
    #for epoch as per threshold
    for i in range(ep):
        # print("for epoch",i)

        lr = lr/(1+i)
        np.random.shuffle(data)
        # data =  data.sample(frac=1,random_state=1)
        for row in range(len(data)):
            y = data[row,0]
            x = data[row,1:]
            # x.pop(0)
            x = np.append(x,1)

            # print("len of x",len(x))
            if np.dot((np.transpose(w)),x)*y <= 1:
                w = ((1-lr)*w) + ((lr*C*y)*x)
            else:
                w = (1-lr)*w

        dict_epoch_accuracy[i] = i,evaluate(w,data),w
        # print("accuracy",accuracy)
            ## calculate the loss/value of the objective function
        # loss = (1/2)*np.dot(np.transpose(w),w)
        # for row in range(len(data)):
        #     y1 = data[row,0]
        #     x1 = data[row,1:]
        #     x1 = np.append(x1, 1)
        #     # loss = round(loss + max(0,(1- (y1 * np.dot(np.transpose(w),x1)))),4)
        #     loss = loss + (C*max(0, (1 - (y1 * np.dot(np.transpose(w), x1)))))
        loss = (1/2)*np.dot(np.transpose(w),w)
        l = 0
        # loss = (1/2)*np.dot(np.transpose(w),w)
        for row in range(len(data)):
            y1 = data[row,0]
            x1 = data[row,1:]
            x1 = np.append(x1, 1)
            # loss = round(loss + max(0,(1- (y1 * np.dot(np.transpose(w),x1)))),4)
            l =  l + (max(0, (1 - (y1 * np.dot(np.transpose(w), x1)))))
        loss = loss + C*l
        # print("loss",loss)
        dict_epoch_loss[i] = loss
        if abs(prev_loss - loss)  < 0.005:
            # print("threshold epoch",i)
            break
        prev_loss = loss

    return dict_epoch_accuracy,dict_epoch_loss





def svm_cross_validation(data,lr,C,ep):
    #initialize weights and bias term
    w = []
    lr_0 = lr
    loss = 0.0
    dict_epoch_accuracy = {}
    dict_epoch_loss = {}
    prev_loss =  float("inf")

    # print(data.shape[1])
    w = np.random.uniform(-0.01,0.01,size=data.shape[1])
    # for i in range(len(data.columns)):
    #     w.append(uniform(-0.01,0.01))
    # print(len(w))
    #for epoch as per threshold
    for i in range(ep):
        # print("for epoch",i)
        accuracy = 0
        lr = lr_0/(1+i)
        np.random.shuffle(data)
        # data =  data.sample(frac=1,random_state=1)
        for row in range(len(data)):
            y = data[row,0]
            x = data[row,1:]
            # x.pop(0)
            x = np.append(x,1)

            # print("len of x",len(x))
            if np.dot((np.transpose(w)),x)*y <= 1:
                w = ((1-lr)*w) + ((lr*C*y)*x)
            else:
                w = (1-lr)*w
                accuracy = accuracy + 1
        dict_epoch_accuracy[i] = i,evaluate(w,data),w
    return dict_epoch_accuracy





def cal_max(d):
    max_w = []
    max_accuracy = -1
    epoch = 0
    for key, value in d.items():
        if value[1] > max_accuracy:
            max_accuracy = value[1]
            max_w = value[2]
            epoch = value[0]
        # if max_accuracy == 0:
    # print(len(max_w))
    return  epoch, max_accuracy,max_w





def evaluate(we, test_data):
    accuracy = 0
    for r in range(len(test_data)):
        y = test_data[r,0]
        x = test_data[r,1:]
        # sample.pop(0)
        x = np.append(x,1)
        # print(len(we))
        # print(len(x))
        prediction = -1 if np.dot(np.transpose(we), x) <= 0 else 1
        # if np.dot((np.transpose(we)),x)*y >= 1:
        if prediction == y:
            accuracy = accuracy + 1
    return (accuracy / len(test_data)) * 100





def crossvalidation(f1,f2,f3,f4,f5):
    lr = [10**0,10**-1,10**-2,10**-3,10**-4,10**-5]
    C = [10**3,10**2,10**1,10**0,10**-1,10**-2]
    best_lr=0
    best_c = 0
    max_acc = 0
    for c in C:
        for l in lr:
            # print("running for learning rate:",l,"C:",c)
            acc = 0
            # run for f1 as test:
            # print("f1 as test")
            frames = [f2, f3, f4, f5]
            train = np.concatenate((f2,f3,f4,f5),axis=0)
            # print("len", len(f2.columns))
            # print("len", len(f3.columns))
            # print("len", len(f4.columns))
            # print("len", len(f5.columns))
            # train = pd.concat(frames)
            # print("len",len(train.columns))
            # dict_acc_w= svm_cross_validation(train, l, c, 20)
            dict_acc_w = svm_cross_validation(train, l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f1)
            # run for f2 as test
            # print("f2 as test")
            frames = [f1, f3, f4, f5]
            train = np.concatenate((f1,f3,f4,f5),axis=0)
            # train = pd.concat(frames)
            dict_acc_w= svm_cross_validation(train, l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w ,f2)
            # run for f3 as test
            # print("f3 as test")
            # frames = [f2, f1, f4, f5]
            # train = pd.concat(frames)
            train = np.concatenate((f1, f2, f4, f5), axis=0)
            dict_acc_w= svm_cross_validation(train, l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f3)
            # run for f4 as test
            # print("f4 as test")
            # frames = [f2, f1, f3, f5]
            # train = pd.concat(frames)
            train = np.concatenate((f1, f3, f2, f5), axis=0)
            dict_acc_w= svm_cross_validation(train, l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f4)
            # run for f5 as test
            # print("f5 as test")
            # frames = [f2, f1, f4, f3]
            # train = pd.concat(frames)
            train = np.concatenate((f1, f3, f4, f2), axis=0)
            dict_acc_w= svm_cross_validation(train, l, c, 20)
            e,a,w = cal_max(dict_acc_w)
            acc = acc + evaluate(w, f5)
            acc = acc/5
            if max_acc<acc:
                max_acc = acc
                best_c =c
                best_lr = l
    print("best parameters", "accuracy",max_acc,"C:",best_c,"lr:",best_lr)
            # dict_margin_lr[(m, lr)] = acc / 5
            # print(dict_margin_lr.keys())
    return max_acc,best_c,best_lr



fold1_svm = new_dataset(weights,bias,fold1)    
fold2_svm = new_dataset(weights,bias, fold2)
fold3_svm = new_dataset(weights,bias, fold3)
fold4_svm = new_dataset(weights,bias, fold4)
fold5_svm = new_dataset(weights,bias, fold5)




a,c,lr = crossvalidation(fold1_svm,fold2_svm,fold3_svm,fold4_svm,fold5_svm)




dict_a_w, dict_loss = svm(train_svm, lr, c, 100)




e, a, w = cal_max(dict_a_w)




evaluate(w, train_svm)




evaluate(w, test_svm)





def evaluate_e(we, test_data):
    accuracy = 0
    p = []
    for r in range(len(test_data)):
        
        x = test_data[r,1:]
        
        x = np.append(x,1)
        # print(len(we))
        # print(len(x))
        prediction = -1 if np.dot(np.transpose(we), x) <= 0 else 1
        p.append(prediction)
        # if np.dot((np.transpose(we)),x)*y >= 1:
#         if prediction == y:
#             accuracy = accuracy + 1
    return p





p= evaluate_e(w, eval_svm)





with open('perceptron_svm_ensemble_tfidf.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["example_id", "label"])
    for i in range(len(p)):
        if p[i]== -1:
            val = 0
        else:
            val = 1
        writer.writerow([i, val])







