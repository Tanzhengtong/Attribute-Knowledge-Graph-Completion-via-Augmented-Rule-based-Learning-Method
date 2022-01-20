from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score, confusion_matrix
from sklearn.datasets import make_classification
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import os
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore")

results=[]

# Get existing predicates in KG
def get_relation_dic(path):
  relation_dic=[]
  with open(path, "r") as f:
      data = f.readlines()
      for line in data:
          odom = line.split()
          entity1=odom[0]
          relation=odom[1]
          entity2=odom[2]
          relation_dic.append(odom)
  return relation_dic

# The number of relation that attribute act as head or tail in a predicate
def find_attribute(index,relation, ishead,relation_dic):
  count=0
  if ishead:
    for pair in relation_dic:
      if pair[0]==index and pair[1]==relation:
        count+=1
  else:
    for pair in relation_dic:
      if pair[2]==index and pair[1]==relation:
        count+=1
  return count

# Filter out relations started with 'inv'
def rule_filter(rule):
  length=len(rule)
  for i in range(length-1,0,-3):
    if rule[i-2].startswith('inv'):
      return True
  return False

# Generate artificial attributes for specific entity
def get_attribute(pair,relations,relation_dic):
    relation = pair[1]
    subj = pair[0]
    obj = pair[2]
    # find the attibutes in which relation(subj,xx)
    subj_match = [find_attribute(subj, i, True,relation_dic) for i in relations]
    # find the attibutes in which relation(xx,obj)
    obj_match = [find_attribute(obj, i, False,relation_dic) for i in relations]
    return subj_match + obj_match

# rules matching
def match(rule, relations,relation_dic):
    # rule head
    head = rule[1:4]
    # rule bodys
    bodys = np.array(rule[4:]).reshape(int(len(rule[4:]) / 3), 3)
    # the index of X and Y in the rule bodys
    index_head = [(0 if np.where(bodys[0] == 'X')[0][0] == 1 else 2),
                  (0 if np.where(bodys[int(len(rule[4:]) / 3) - 1] == 'Y')[0][0] == 1 else 2)]

    # matching bodys
    results = []
    for i in range(len(bodys)):
        if i != 0 and (results == [] or len(results) <= 5):
            return None, None, 0, 0, None, None
        if i == 0:
            results = matchhelp(None, bodys[i][0], None, None, relation_dic)
        else:
            # body format'niece', 'X', 'A', 'sister', 'A', 'Y']
            # rule format in relation_dic ['143', 'niece', '340']
            if bodys[i - 1][2] == bodys[i][1]:
                results = matchhelp(2, bodys[i][0], 0, results, relation_dic)
            elif bodys[i - 1][1] == bodys[i][1]:
                results = matchhelp(0, bodys[i][0], 0, results, relation_dic)
            elif bodys[i - 1][2] == bodys[i][2]:
                results = matchhelp(2, bodys[i][0], 2, results, relation_dic)
            elif bodys[i - 1][1] == bodys[i][2]:
                results = matchhelp(0, bodys[i][0], 2, results, relation_dic)
    # features
    features = []
    # labels
    Y = []
    # recursive head matching pairs
    headmatching = []
    # numerator of SC
    numerator = 0
    for result in results:
        feature = []
        bodys = np.array(result).reshape(int(len(result) / 3), 3)
        for body in bodys:
            feature.extend(get_attribute(body,relations,relation_dic))
        features.append(feature)

        # check if the pair match the rule head
        if [bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[
            1]]] in relation_dic:  # and [bodys[0][index_head[0]],head[0],bodys[int(len(result)/3)-1][index_head[1]]] not in bodys:
            numerator += 1
            Y.append(1)
            headmatching.append([bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]])
        else:
            Y.append(0)
            headmatching.append([])
            # table head(attribute names for the function output)
    tablehead = []
    for body in bodys:
        tablehead += (body[1] + '_is_' + pd.Series(relations) + '_of').tolist()
        tablehead += (body[2] + '_has_' + pd.Series(relations)).tolist()
    # print(tablehead)
    # print(len(tablehead))
    # print()
    denominator = len(results)
    #   dataframe=pd.DataFrame(np.array(features),columns=tablehead)
    dataframe = np.array(features)
    return dataframe, Y, numerator, denominator, tablehead, headmatching


def matchhelp(former, relation, current, candidates, relation_dic):
    results = []
    if former is None and current is None:
        for r in relation_dic:
            if r[1] == relation:
                results.append(r)
        return results
    else:
        curlen = len(candidates[0])
        for cur in candidates:
            candidates.remove(cur)
            for r in relation_dic:
                if r[current] == cur[curlen - 3 + former] and r[1] == relation and r != cur:
                    # print(current,former)
                    # print(cur+r)
                    results.append(cur + r)
        return results

# extract rules from rule-based system
def rule_extract(filepath):
    rules = []
    with open(filepath, "r") as f:
        data = f.readlines()
        for rule in data:
            rules.append(rule)
    return rules

# check the original rule quality
def rule_check(rule):#,min_threshold,max_threshold):
    SC = float(rule[0])
    rule_head = rule[1:4]
    return rule_head[1] == 'X' and rule_head[2] == 'Y' and (SC > 0.02 and SC <=0.3)

# get head coverage HC
def get_headcoverage(rule,support,relation_dic):
    rule_head_relation=rule[1]
    size=sum([r[1]==rule_head_relation for r in relation_dic])
    head_coverage=support/size if size!=0 else 0
    return head_coverage

def augmented_rules(i,relations,relation_dic,rules):
    added_index=[15941,18456,3464,15873,19177,19323,19486,19288,16824,\
                 18397,18261,17750,15913,17098,16669,\
                 3234,14409,14197,14958,13887,98,176,\
                 15948,12594,18221,16567,10908,15554,13833,\
                 7187,7486,17248,5787,15163,15161,15160,15159]
    if i in added_index:
        return 0
    print(i)
    cur_rule = re.sub(r'\t|\n|<--|\(|\)|,', ' ', rules[i]).split()
    cur_rule=cur_rule[2:6]+cur_rule[7:]
    # rules have to meet some conditions: SC is greater than a threshold e.g.0.5)
    if not rule_check(cur_rule):
        # print("No. %d" % i, " Original rule does not meets the requirements")
        return 0
    print("No. %d" % i, " Original rule meets the requirements")
    # Data preparation
    X,Y,numerator,denominator,attributehead,headmatching=match(cur_rule,relations,relation_dic)
    # DT construction requirement: numerator is at least greater than 2 and SC cannot be 1
    if X is None or Y is None or numerator<=2 or denominator-numerator<=2 or numerator==denominator:
        return 0
    print("No. %d" % i, " Rule matching meets the requirements")
    print(numerator,denominator,np.array(X).shape,np.array(Y).shape)
    #   print("The index of rule is ",i, " The rule is ",cur_rule)

    # Calculate the head coverage
    head_coverage = get_headcoverage(cur_rule, numerator,relation_dic)
    if head_coverage < 0.01:
        return 0
    # resampling for balanced data set
    if np.array(X).shape[0]<np.array(X).shape[1]:
        x_sample, y_sample = SMOTE(k_neighbors=2).fit_resample(np.array(X), np.array(Y).ravel())
    else:
        x_sample, y_sample = X,Y
    # data split
    x_train, x_test, y_train, y_test = \
              train_test_split(x_sample, y_sample, test_size=0.2,shuffle=True)
    #model construction
    model = tree.DecisionTreeClassifier(splitter='random',class_weight='balanced')
    model.fit(x_train, y_train)

    #   plt.figure(figsize=(15,15))  # set plot size (denoted in inches)
    #   cn=['Nohead',"Head"]
    #   # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    #   tree.plot_tree(model,max_depth=3,feature_names = attributehead, class_names=cn,filled = True,fontsize=8)
    # test on the original data set without any resampling
    #   print("original confidence score: ",numerator,denominator,numerator/denominator)

    pred_y=model.predict(X)
    tn, fp, fn, tp=confusion_matrix(Y,pred_y).ravel()
    #   print("current confidence score: ",'tp ',tp,'fp ',fp,'tp+fp ',tp+fp,(tp)/(tp+fp))
    # ignore those with no SC improvement
    if numerator/denominator > tp/(tp+fp):
        return 0
    # results
    unq = np.array([x + 2*y for x, y in zip(pred_y, Y)])
    tpindex = np.array(np.where(unq ==3)).tolist()[0]
    #   print(headmatching[j])
    #   print([str(tp),str(tp+fp)])
    #   print(rules[i])
    result=[[re.sub('\n', '', rules[i])]+headmatching[j]+[str(tp/(tp+fp))] for j in tpindex]
    with open('rules/augmented_rule-valid-test','a') as f:
        for r in result:
            f.write("\t".join(r)+'\r\n')
    print("Current index "+str(i)+" Add "+str(len(result))+" Records")
    return 1


# Step 2: Define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)

if __name__ == '__main__':
    # read rules
    rules=rule_extract("rules/alpha-10")
    # reading predicates
    relation_dic = []
    # relation_dic.extend(get_relation_dic("data 3/family/train.txt"))
    # print(len(relation_dic))
    relation_dic.extend(get_relation_dic("data 3/FB15-237/valid.txt"))
    relation_dic.extend(get_relation_dic("data 3/FB15-237/test.txt"))
    print(len(relation_dic))

    # reading relations
    relations = set()
    for pair in relation_dic:
        relations.add(pair[1])
    relations = list(relations)

    # 533.455118894577s 0-200
    # 544.513.6151313782ms 0-200
    # 527.595942735672s 0-200

    # T1 = time.time()
    # for i in range(0,50):
    #     augmented_rules(i,relations,relation_dic,rules)
    # T2 = time.time()
    # print('The program runs about :%s s' % (T2 - T1))


    # T1 = time.time()
    # # Step 1: Init multiprocessing.Pool()
    # pool = mp.Pool(mp.cpu_count())
    # # Step 2: `pool.apply` the `augmented_rules()`
    # results = [pool.apply(augmented_rules, args=(row,relations,relation_dic,rules)) for row in range(0,50)]
    # # # Step 3: Don't forget to close
    # pool.close()
    # # print(results)
    # T2 = time.time()
    # print('The program runs about :%s s' % (T2 - T1 ))


    T1 = time.time()
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 3: Use loop to parallelize
    for i in range(0,len(rules)):
        pool.apply_async(augmented_rules, args=(i,relations,relation_dic,rules), callback=collect_result)
    # Step 4: Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    T2 = time.time()
    print('The program runs about :%s s' % (T2 - T1))
    print(sum(results))

