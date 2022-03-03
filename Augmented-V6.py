"""
V6: support specific rules
"""
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
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


# Get attributes triples in KG
def get_attribute_dic(path):
    attribite_dic = {}
    if not os.path.isfile(path + "/train-attr.txt") or not os.path.isfile(path + "/test-attr.txt"):
        return None
    with open(path + "/train-attr.txt", "r") as f:
        data = f.readlines()
        for line in data:
            odom = line.split()
            entity = odom[0]
            value = odom[1]
            attribute = odom[2]
            attribite_dic[entity + attribute] = value
    with open(path + "/test-attr.txt", "r") as f:
        data = f.readlines()
        for line in data:
            odom = line.split()
            entity = odom[0]
            value = odom[1]
            attribute = odom[2]
            attribite_dic[entity + attribute] = value
    return attribite_dic


# Get attributes name in KG
def get_attributes(path):
    if not os.path.isfile(path):
        return None
    attributes = set()
    with open(path, "r") as f:
        data = f.readlines()
        for line in data:
            attributes.add(line.split()[0])
    return list(attributes)


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
    subj = pair[0]
    obj = pair[2]
    # find the attibutes in which relation(subj,xx)
    subj_match = [find_attribute(subj, i, True, relation_dic) for i in relations]
    # find the attibutes in which relation(xx,obj)
    obj_match = [find_attribute(obj, i, False, relation_dic) for i in relations]
    return subj_match + obj_match


# Categorical attribute to discrete type
def categorical_to_discrete(path):
    if not os.path.isfile(path):
        return None
    attribute_mapping_dic = {}
    with open(path, "r") as f:
        data = f.readlines()
        cur_attribute = None
        for line in data:
            odom = line.split()
            if odom is None:
                continue
            elif len(odom) > 1 and odom[1].isdigit():
                cur_attribute = odom[0]
                attribute_mapping_dic[cur_attribute] = {}
                attribute_mapping_dic[cur_attribute]["None"] = 0
            else:
                for i in range(0, len(odom)):
                    attribute_mapping_dic[cur_attribute][odom[i]] = i + 1
    # print(attribute_mapping_dic.get("common.topic.notable_types").get("None"))
    return attribute_mapping_dic


# Extra attribute from AKG!!
def get_extra_attribute(bodys, attributes, attribite_dic, attribute_mapping_dic):
    entities = set()
    for body in bodys:
        entities.add(body[0])
        entities.add(body[2])
    entities = list(entities)
    table_head_list = []
    attribute_list = []
    for i in range(0, len(entities)):
        entity = entities[i]
        attribute_list.extend(
            [attribute_mapping_dic.get(attribute).get(attribite_dic.get(entity + attribute, "None")) for attribute in
             attributes])
        table_head_list += ("Entity " + str(i) + " " + pd.Series(attributes)).tolist()
    return attribute_list, table_head_list


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
    return sc >= min_threshold and sc <= max_threshold and len(rule) > 4


# check if the rule is closed and connected rule
def connected_closed_rule_check(rule):
    rule_head = rule[1:4]
    return rule_head[1] == 'X' and rule_head[2] == 'Y'


# get head coverage HC
def get_headcoverage(rule, support, relation_dic):
    rule_head_relation = rule[1]
    size = sum([r[1] == rule_head_relation for r in relation_dic])
    head_coverage = support / size if size != 0 else 0
    return head_coverage


"""   Dataframe simplification"""


# Rrmove all the irrelevant attributes
def simplify_datainput(tablehead, dataframe):
    transpose = np.transpose(dataframe)
    # transpose= dataframe.T
    shape = np.shape(dataframe)
    id_to_delete = []

    for i in range(0, shape[1]):
        if (transpose[i] == np.zeros((shape[0],), dtype=int)).all():
            id_to_delete.append(i)

    new_tablehead = []
    new_dataframe = []
    for i in range(0, shape[1]):
        if i not in id_to_delete:
            new_tablehead.append(tablehead[i])
            new_dataframe.append(transpose[i])
    return new_tablehead, np.transpose(new_dataframe)


"""    Rule matching module   """


# # rules matching
# def match(rule, relations, relation_dic, attributes=None, attribite_dic=None, attribute_mapping_dic=None):
#     # rule head
#     head = rule[1:4]
#     # rule bodys
#     bodys = np.array(rule[4:]).reshape(int(len(rule[4:]) / 3), 3)
#     # the index of X and Y in the rule bodys
#     index_head = [(0 if np.where(bodys[0] == 'X')[0][0] == 1 else 2),
#                   (0 if np.where(bodys[int(len(rule[4:]) / 3) - 1] == 'Y')[0][0] == 1 else 2)]
#
#     # matching bodys
#     results = []
#     for i in range(len(bodys)):
#         #  total number of instance should be at least 4 since each class should have at least two instance
#         if i != 0 and (results == [] or len(results) <= 3):
#             return None, None, 0, 0, None, None
#         if i == 0:
#             results = matchhelp(None, bodys[i][0], None, None, relation_dic)
#         else:
#             # body format 'niece', 'X', 'A', 'sister', 'A', 'Y'
#             # rule format in relation_dic ['143', 'niece', '340']
#             if bodys[i - 1][2] == bodys[i][1]:
#                 results = matchhelp(2, bodys[i][0], 0, results, relation_dic)
#             elif bodys[i - 1][1] == bodys[i][1]:
#                 results = matchhelp(0, bodys[i][0], 0, results, relation_dic)
#             elif bodys[i - 1][2] == bodys[i][2]:
#                 results = matchhelp(2, bodys[i][0], 2, results, relation_dic)
#             elif bodys[i - 1][1] == bodys[i][2]:
#                 results = matchhelp(0, bodys[i][0], 2, results, relation_dic)
#     # total number of instance should be at least 4 since each class should have at least two instance
#     if results == [] or len(results) <= 3:
#         return None, None, 0, 0, None, None
#     # features
#     features = []
#     # labels
#     Y = []
#     # recursive head matching pairs
#     headmatching = []
#     # numerator of SC
#     numerator = 0
#     # the head of the feature tables
#     tablehead = []
#     for body in bodys:
#         tablehead += (body[1] + '_is_' + pd.Series(relations) + '_of').tolist()
#         tablehead += (body[2] + '_has_' + pd.Series(relations)).tolist()
#
#     for result in results:
#         feature = []
#         bodys = np.array(result).reshape(int(len(result) / 3), 3)
#         for body in bodys:
#             feature.extend(get_attribute(body, relations, relation_dic))
#         # !! Extra features from AKG
#         if attribite_dic is not None and attributes is not None:
#             extra_feature, extra_tablehead = get_extra_attribute(bodys, attributes, attribite_dic,
#                                                                  attribute_mapping_dic)
#             feature.extend(extra_feature)
#             tablehead.extend(extra_tablehead)
#
#         features.append(feature)
#         # check if the pair match the rule head
#         if [bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[
#             1]]] in relation_dic:  # and [bodys[0][index_head[0]],head[0],bodys[int(len(result)/3)-1][index_head[1]]] not in bodys:
#             numerator += 1
#             Y.append(1)
#             headmatching.append([bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]])
#         else:
#             Y.append(0)
#             headmatching.append([])
#             # table head(attribute names for the function output)
#
#     # !! Extra Attribute from AKG
#     # dataframe construction
#     # dataframe = np.array(features)
#     tablehead, dataframe = simplify_datainput(tablehead, features)
#     denominator = len(results)
#     return dataframe, Y, numerator, denominator, tablehead, headmatching


# rules matching
def match(rule, is_connected_closed, relations, relation_dic, attributes=None, attribite_dic=None,
               attribute_mapping_dic=None):
    # rule head
    head = np.array(rule[1:4])
    # rule bodys
    bodys = np.array(rule[4:]).reshape(int(len(rule[4:]) / 3), 3)
    # the index of X and Y in the rule bodies
    index_head = [(0 if np.where(bodys[0] == 'X')[0][0] == 1 else 2), (
        0 if np.where(bodys[int(len(rule[4:]) / 3) - 1] == 'Y')[0][0] == 1 else 2)] if is_connected_closed else [
        np.where((head == 'Y') | (head == 'X'))[0][0],
        np.where((bodys[int(len(rule[4:]) / 3) - 1] == 'Y') | (bodys[int(len(rule[4:]) / 3) - 1] == 'X'))[0][0]]
    is_X_in_head = True if ('X' in head and 'Y' not in head) else False
    # matching bodys
    results = []
    for i in range(len(bodys)):
        # base case
        # total number of instance should be at least 4 since each class should have at least two instance
        if i != 0 and (results == [] or len(results) <= 3):
            return None, None, 0, 0, None, None
        if i == 0:
            results = matchhelp(None, None, bodys[i], None, relation_dic)
        else:
            # body format 'niece', 'X', 'A', 'sister', 'A', 'Y'
            # rule format in relation_dic ['143', 'niece', '340']
            if bodys[i - 1][2] == bodys[i][1]:
                results = matchhelp(2, 0, bodys[i], results, relation_dic)
            elif bodys[i - 1][1] == bodys[i][1]:
                results = matchhelp(0, 0, bodys[i], results, relation_dic)
            elif bodys[i - 1][2] == bodys[i][2]:
                results = matchhelp(2, 2, bodys[i], results, relation_dic)
            elif bodys[i - 1][1] == bodys[i][2]:
                results = matchhelp(0, 2, bodys[i], results, relation_dic)
    # total number of instance should be at least 4 since each class should have at least two instance
    if results == [] or len(results) <= 3:
        return None, None, 0, 0, None, None
    # features
    features = []
    # labels
    Y = []
    # recursive head matching pairs
    headmatching = []
    # numerator of SC
    numerator = 0
    # the head of the feature tables
    tablehead = []
    for body in bodys:
        tablehead += (body[1] + '_is_' + pd.Series(relations) + '_of').tolist()
        tablehead += (body[2] + '_has_' + pd.Series(relations)).tolist()
    for result in results:
        feature = []
        bodys = np.array(result).reshape(int(len(result) / 3), 3)
        for body in bodys:
            feature.extend(get_attribute(body, relations, relation_dic))
        # !! Extra features from AKG
        if attribite_dic is not None and attributes is not None:
            extra_feature, extra_tablehead = get_extra_attribute(bodys, attributes, attribite_dic,
                                                                 attribute_mapping_dic)
            feature.extend(extra_feature)
            tablehead.extend(extra_tablehead)
        features.append(feature)
        # check if the pair match the rule head
        if is_connected_closed:
            if [bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]] in relation_dic:
                numerator += 1
                Y.append(1)
                headmatching.append([bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]])
            else:
                Y.append(0)
                headmatching.append([])
        elif not is_connected_closed and is_X_in_head and index_head[0] == 1:
            if [bodys[int(len(result) / 3) - 1][index_head[1]], head[0], head[2]] in relation_dic:
                numerator += 1
                Y.append(1)
                headmatching.append([bodys[int(len(result) / 3) - 1][index_head[1]], head[0], head[2]])
            else:
                Y.append(0)
                headmatching.append([])
        elif not is_connected_closed and not is_X_in_head and index_head[0] == 2:
            if [head[1], head[0], bodys[int(len(result) / 3) - 1][index_head[1]], head[0]] in relation_dic:
                numerator += 1
                Y.append(1)
                headmatching.append([head[1], head[0], bodys[int(len(result) / 3) - 1][index_head[1]], head[0]])
            else:
                Y.append(0)
                headmatching.append([])

    # !! Extra Attribute from AKG
    tablehead, dataframe = simplify_datainput(tablehead, features)
    denominator = len(results)
    return dataframe, Y, numerator, denominator, tablehead, headmatching


# check if the entity is a general entity in the form of "X" for example
def is_general_entity(entity):
    return len(entity) == 1


def matchhelp(former, current, body, candidates, relation_dic):
    """
    :param former: previous connected entity index if exist
    :param current: current connected entity index if exist
    :param body: current body
    :param candidates: results series that match the conditions
    :param relation_dic: relation dictionary
    :return: results series that match the current conditions
    """
    relation, entity1, entity2 = body[0], body[1], body[2]
    results = []
    # first body:
    if former is None and current is None:
        # if the former entity is in specific form
        if not is_general_entity(entity1):
            for r in relation_dic:
                if r[1] == relation and r[0] == entity1:
                    results.append(r)
        # if the latter entity is in specific form
        elif not is_general_entity(body[2]):
            for r in relation_dic:
                if r[1] == relation and r[2] == entity2:
                    results.append(r)
        # if all the entities are in general form
        else:
            for r in relation_dic:
                if r[1] == relation:
                    results.append(r)
    # other than first body:
    else:
        curlen = len(candidates[0])
        # if the former entity is in specific form
        if not is_general_entity(entity1):
            for cur in candidates:
                candidates.remove(cur)
                for r in relation_dic:
                    if r[current] == cur[curlen - 3 + former] and r[1] == relation and r[0] == entity1 and r != cur:
                        results.append(cur + r)
        # if the latter entity is in specific form
        elif not is_general_entity(entity2):
            for cur in candidates:
                candidates.remove(cur)
                for r in relation_dic:
                    if r[current] == cur[curlen - 3 + former] and r[1] == relation and r[2] == entity2 and r != cur:
                        results.append(cur + r)
        # if all the entities are in general form
        else:
            for cur in candidates:
                candidates.remove(cur)
                for r in relation_dic:
                    if r[current] == cur[curlen - 3 + former] and r[1] == relation and r != cur:
                        results.append(cur + r)
    return results

def augmented_rules(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage_threshold, support_threshold,
                    attributes=None, attribite_dic=None, attribute_mapping_dic=None):
    cur_rule = re.sub(r'\t|\n|<--|\(|\)|,', ' ', rules[i]).split()
    cur_rule = cur_rule[2:6] + cur_rule[7:]
    # rules have to meet some conditions: SC is greater than a threshold e.g.0.5)
    if not rule_pre_check(cur_rule, sc_min, sc_max):
        # print("No. %d" % i, " Original rule does not meet the pre-conditions")
        return 0
    # print("No. %d" % i, " Original rule meets the pre-conditions")
    is_connected_closed = connected_closed_rule_check(cur_rule)
    # Data preparation
    X, Y, numerator, denominator, attributehead, headmatching = match(cur_rule, is_connected_closed, relations,
                                                                           relation_dic,
                                                                           attributes,
                                                                           attribite_dic, attribute_mapping_dic)
    # DT construction requirement:
    # support is at least greater than support_threshold and the number in each class should be greater than 2
    if X is None or Y is None or numerator <= support_threshold or denominator - numerator < 2 or numerator == denominator:
        return 0
    print("No. %d " % i, "connected ", is_connected_closed, numerator, denominator, np.array(Y).shape,
          np.array(X).shape)

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

    # Decision Tree Model-Cart default=”gini”
    # model = tree.DecisionTreeClassifier(splitter='random', class_weight='balanced')
    # model.fit(x_train, y_train)

    # Decision Tree Model-ID3
    # model = tree.DecisionTreeClassifier(splitter='random', class_weight='balanced',criterion='entropy')
    # model.fit(x_train, y_train)
    #
    # Random Forest Model
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(x_train, y_train)
    #
    # # Logistic Regression
    # model = LogisticRegression(penalty='l2')
    # model.fit(x_train, y_train)
    #
    # # Gradient Boosting Classifier
    # model = GradientBoostingClassifier(n_estimators=200)
    # model.fit(x_train, y_train)
    #
    # #AdaBoost Classifier
    # model = AdaBoostClassifier()
    # model.fit(x_train, y_train)

    # Model result visualization
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
    print("Current index ", "is_connected ", is_connected_closed, str(i), "Add ", len(result), " Records")
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

    # Attributes
    attribite_dic = get_attribute_dic("data 3/" + data_set)
    attributes = get_attributes("data 3/" + data_set + "/attribute2id.txt")
    attribute_mapping_dic = categorical_to_discrete("data 3/" + data_set + "/attribute_val.txt")

    print("The number of facts: ", len(relation_dic))
    print("The number of relations: ", len(relations))
    print("The number of rules: ", len(rules))
    if attribite_dic is not None and attributes is not None:
        print("The number of attribute dic: ", len(attribite_dic))
        print("The number of attribute type: ", len(attributes))

    # Asynchronous parallel computing
    T1 = time.time()
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: Use loop to parallelize
    for i in range(0, len(rules)):
        pool.apply_async(augmented_rules,
                         args=(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage, support_threshold,
                               attributes, attribite_dic, attribute_mapping_dic),
                         callback=collect_result)

    # Step 4: Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    T2 = time.time()
    print('The DT program runs about :%s s' % (T2 - T1), "Got %s augmented rules" % (sum(results)))

    # # sequential computing
    # T1 = time.time()
    # for i in range(0,len(rules)):
    #     augmented_rules(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage, support_threshold, attributes,
    #                     attribite_dic, attribute_mapping_dic)
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
