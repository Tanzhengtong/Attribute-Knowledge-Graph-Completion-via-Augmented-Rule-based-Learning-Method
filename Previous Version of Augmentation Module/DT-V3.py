"""
V3: Simplify the attribute/ input for classifier
simplify_datainput
"""

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
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
import sys

warnings.filterwarnings("ignore")
results = []
"""    Data Load    """


# Get existing predicates in KG
def get_relation_dic(path):
    relation_dic = []
    with open(path, "r") as f:
        data = f.readlines()
        for line in data:
            odom = line.split()
            relation_dic.append(odom)
    return relation_dic


# extract rules from rule-based system
def rule_extract(filepath):
    rules = []
    with open(filepath, "r") as f:
        data = f.readlines()
        for rule in data:
            rules.append(rule)
    return rules


"""    Attributes Generation   """


# The number of relation that attribute act as head or tail in a predicate
def find_attribute(index, relation, ishead, relation_dic):
    count = 0
    if ishead:
        for pair in relation_dic:
            if pair[0] == index and pair[1] == relation:
                count += 1
    else:
        for pair in relation_dic:
            if pair[2] == index and pair[1] == relation:
                count += 1
    return count


# Generate artificial attributes for specific entity
def get_attribute(pair, relations, relation_dic):
    relation = pair[1]
    subj = pair[0]
    obj = pair[2]
    # find the attibutes in which relation(subj,xx)
    subj_match = [find_attribute(subj, i, True, relation_dic) for i in relations]
    # find the attibutes in which relation(xx,obj)
    obj_match = [find_attribute(obj, i, False, relation_dic) for i in relations]
    return subj_match + obj_match


"""    Rule filters   """
# Filter out relations started with 'inv'
def rule_filter(rule):
    length = len(rule)
    for i in range(length - 1, 0, -3):
        if rule[i - 2].startswith('inv'):
            return True
    return False


# check the original rule quality
def rule_pre_check(rule, min_threshold, max_threshold):
    sc = float(rule[0])
    rule_head = rule[1:4]
    return rule_head[1] == 'X' and rule_head[2] == 'Y' and (sc >= min_threshold and sc <= max_threshold)


# get head coverage HC
def get_headcoverage(rule, support, relation_dic):
    rule_head_relation = rule[1]
    size = sum([r[1] == rule_head_relation for r in relation_dic])
    head_coverage = support / size if size != 0 else 0
    return head_coverage


"""   Dataframe simplification"""
def simplify_datainput(tablehead, dataframe):
    transpose = np.transpose(dataframe)
    shape = np.shape(dataframe)
    id_to_delete = []

    # print("Here is simplify")
    # print(shape)
    # print(len(transpose))
    # print(shape[0])
    for i in range(0, shape[1]):

        if (transpose[i] == np.zeros((shape[0],), dtype=int)).all() or (
                transpose[i] == np.ravel(np.full([1, shape[0]], np.nan))).all():
            id_to_delete.append(i)
    # print(id_to_delete)
    new_tablehead = []
    new_dataframe = []
    for i in range(0, shape[1]):
        if i not in id_to_delete:
            new_tablehead.append(tablehead[i])
            new_dataframe.append(transpose[i])
    # print(np.transpose(new_dataframe))
    # print(new_tablehead)
    return new_tablehead, np.transpose(new_dataframe)


"""    Rule matching module   """

# rules matching
def match(rule, relations, relation_dic):
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
        # standard confidence numerator should be at least 5
        if i != 0 and (results == [] or len(results) <= 5):
            return None, None, 0, 0, None, None
        if i == 0:
            results = matchhelp(None, bodys[i][0], None, None, relation_dic)
        else:
            # body format 'niece', 'X', 'A', 'sister', 'A', 'Y']
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
            feature.extend(get_attribute(body, relations, relation_dic))
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
    # the head of the feature tables
    tablehead = []
    for body in bodys:
        tablehead += (body[1] + '_is_' + pd.Series(relations) + '_of').tolist()
        tablehead += (body[2] + '_has_' + pd.Series(relations)).tolist()

    #   dataframe=pd.DataFrame(np.array(features),columns=tablehead)
    dataframe = np.array(features)
    tablehead, dataframe = simplify_datainput(tablehead, dataframe)
    denominator = len(results)
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


def augmented_rules(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage_threshold, support_threshold):
    cur_rule = re.sub(r'\t|\n|<--|\(|\)|,', ' ', rules[i]).split()
    cur_rule = cur_rule[2:6] + cur_rule[7:]
    # rules have to meet some conditions: SC is greater than a threshold e.g.0.5)
    if not rule_pre_check(cur_rule, sc_min, sc_max):
        # print("No. %d" % i, " Original rule does not meets the requirements")
        return 0
    print("No. %d" % i, " Original rule meets the pre-conditions")
    # Data preparation
    X, Y, numerator, denominator, attributehead, headmatching = match(cur_rule, relations, relation_dic)
    # DT construction requirement:
    # support is at least greater than 10 and
    # SC cannot be 1
    print(numerator, denominator, np.array(X).shape, np.array(Y).shape)
    if X is None or Y is None or numerator <= support_threshold or denominator - numerator <= 2 or numerator == denominator:
        return 0
    # print(numerator,denominator,np.array(X).shape,np.array(Y).shape)
    #   print("The index of rule is ",i, " The rule is ",cur_rule)

    # Calculate the head coverage
    head_coverage = get_headcoverage(cur_rule, numerator, relation_dic)
    if head_coverage < head_coverage_threshold:
        return 0
    # print("No. %d" % i, " Rule matching meets the requirements")
    # resampling for balanced data set
    if np.array(X).shape[0] < np.array(X).shape[1]:
        x_sample, y_sample = SMOTE(k_neighbors=2).fit_resample(np.array(X), np.array(Y).ravel())
    else:
        x_sample, y_sample = X, Y
    # data split
    x_train, x_test, y_train, y_test = \
        train_test_split(x_sample, y_sample, test_size=0.2, shuffle=True)
    # model construction
    model = tree.DecisionTreeClassifier(splitter='random', class_weight='balanced')
    model.fit(x_train, y_train)

    #   plt.figure(figsize=(15,15))  # set plot size (denoted in inches)
    #   cn=['Nohead',"Head"]
    #   # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    #   tree.plot_tree(model,max_depth=3,feature_names = attributehead, class_names=cn,filled = True,fontsize=8)
    # test on the original data set without any resampling
    #   print("original confidence score: ",numerator,denominator,numerator/denominator)

    pred_y = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(Y, pred_y).ravel()
    #   print("current confidence score: ",'tp ',tp,'fp ',fp,'tp+fp ',tp+fp,(tp)/(tp+fp))
    # ignore those with no SC improvement
    if numerator / denominator > tp / (tp + fp):
        return 0
    # results
    unq = np.array([x + 2 * y for x, y in zip(pred_y, Y)])
    tpindex = np.array(np.where(unq == 3)).tolist()[0]
    result = [[re.sub('\n', '', rules[i])] + headmatching[j] + [str(tp / (tp + fp))] for j in tpindex]
    with open('rules/augmented_rule-valid-test', 'a') as f:
        for r in result:
            f.write("\t".join(r) + '\r\n')
    print("Current index " + str(i) + " Add " + str(len(result)) + " Records")
    return 1


# Step 3: Define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)


if __name__ == '__main__':
    paramters = sys.argv[1:]
    data_set, sc_min, sc_max, head_coverage, support_threshold, learning_time = \
        paramters[0], float(paramters[1]), float(paramters[2]), float(paramters[3]), int(paramters[4]), str(
            paramters[5])
    print("Current hyper-parameters: ", data_set, sc_min, sc_max, head_coverage, support_threshold, learning_time)
    # read rules
    rules = rule_extract("rules/alpha-" + learning_time)

    # reading predicates
    relation_dic = []
    # relation_dic.extend(get_relation_dic("data 3/"+data_set+"/train.txt"))
    relation_dic.extend(get_relation_dic("data 3/" + data_set + "/valid.txt"))
    relation_dic.extend(get_relation_dic("data 3/" + data_set + "/test.txt"))
    # reading relations
    relations = set()
    for pair in relation_dic:
        relations.add(pair[1])
    relations = list(relations)

    print("The number of facts: ", len(relation_dic))
    print("The number of relations: ", len(relations))
    print("The number of rules: ", len(rules))

    # Asynchronous parallel computing
    T1 = time.time()
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: Use loop to parallelize
    # for i in range(0, len(rules)):
    for i in range(250,400):
        pool.apply_async(augmented_rules,
                         args=(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage, support_threshold),
                         callback=collect_result)

    # Step 4: Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    T2 = time.time()
    print('The DT program runs about :%s s' % (T2 - T1), "Got %s augmented rules" % (sum(results)))

    # sequential computing
    # T1 = time.time()
    # for i in range(0,50):
    #     augmented_rules(i,relations,relation_dic,rules)
    # T2 = time.time()
    # print('The program runs about :%s s' % (T2 - T1))

    # synchronous parallel computing
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
