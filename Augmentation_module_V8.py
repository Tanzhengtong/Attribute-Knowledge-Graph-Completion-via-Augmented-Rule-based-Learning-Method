"""
V8: Introduce the Grid Search
"""
from category_encoders import TargetEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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
import pandas as pd
import operator
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

# extract entities from
def entity_inspection(dataset):
    entities = set()
    for pair in dataset:
        entities.add(pair[0])
        entities.add(pair[2])
    return list(entities)


# Get attributes triples in KG
def get_attribute_dic(path, contains_numerical_attribute):
    attribite_dic = {}
    if not os.path.isfile(path + "/train-attr.txt") and not os.path.isfile(path + "/test-attr.txt"):
        return None
    if os.path.isfile(path + "/train-attr.txt"):
        with open(path + "/train-attr.txt", "r") as f:
            data = f.readlines()
            for line in data:
                odom = line.split()
                entity = odom[0]
                value = odom[1]
                attribute = odom[2]
                if contains_numerical_attribute:
                    attribite_dic[entity + attribute] = float(value)
                else:
                    attribite_dic[entity + attribute] = value
    if os.path.isfile(path + "/test-attr.txt"):
        with open(path + "/test-attr.txt", "r") as f:
            data = f.readlines()
            for line in data:
                odom = line.split()
                entity = odom[0]
                value = odom[1]
                attribute = odom[2]
                if contains_numerical_attribute:
                    attribite_dic[entity + attribute] = float(value)
                else:
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
def get_extra_attribute(bodys, attributes, attribite_dic, contains_numerical_attribute):  # , attribute_mapping_dic):
    entities = set()
    for body in bodys:
        entities.add(body[0])
        entities.add(body[2])
    entities = list(entities)
    table_head_list = []
    attribute_list = []
    if contains_numerical_attribute == 0:
        for i in range(0, len(entities)):
            entity = entities[i]
            attribute_list.extend([attribite_dic.get(entity + attribute, "None") for attribute in attributes])
            table_head_list += ("Entity " + str(i) + " " + pd.Series(attributes)).tolist()
    else:
        for i in range(0, len(entities)):
            entity = entities[i]
            attribute_list.extend([attribite_dic.get(entity + attribute, 0) for attribute in attributes])
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


"""   Dataframe simplification/ preprocessing  """

# target encoding for categorical values
def target_encoding(extra_feature, Y):
    enc = TargetEncoder()
    extra_feature = enc.fit_transform(extra_feature, Y)
    return extra_feature

# Remove all the irrelevant attributes
def simplify_datainput(tablehead, dataframe):
    transpose = np.transpose(dataframe)
    shape = np.shape(dataframe)
    id_to_delete = []
    for i in range(0, shape[1]):
        # all columns are zeros or
        if (transpose[i] == np.zeros((shape[0],), dtype=int)).all() or (
                (transpose[i] - transpose[i][0]) == np.zeros((shape[0],), dtype=int)).all():
            id_to_delete.append(i)
    new_tablehead = []
    new_dataframe = []
    for i in range(0, shape[1]):
        if i not in id_to_delete:
            new_tablehead.append(tablehead[i])
            new_dataframe.append(transpose[i])
    return new_tablehead, np.transpose(new_dataframe)


"""    Rule matching module   """

# find the triplets set that satisfy the rule
def match(rule, is_connected_closed, relations, relation_dic, attributes=None,
          attribite_dic=None, contains_numerical_attribute=0):
    '''

    :param rule: rules from rule-based system
    :param is_connected_closed: Boolean parameter that check if the rule is CCR
    :param relations: relational triplets in KG
    :param relation_dic: relations in KG
    :param attributes: attributive triplets if AKG
    :param attribite_dic: attributes in AKG
    :param contains_numerical_attribute: Boolean parameter that check if the AKG contain numerical attribute
    :return: dataframe(atrribute matrix), Y (class label), numerator (of SC), denominator (of SC), tablehead (attribute names), headmatching (predicates that match the rule head)
    '''
    # rule head
    head = np.array(rule[1:4])
    # rule bodys
    bodys = np.array(rule[4:]).reshape(int(len(rule[4:]) / 3), 3)
    # the index of X and Y in the rule bodies
    index_head = [(0 if np.where(bodys[0] == 'X')[0][0] == 1 else 2), (
        0 if np.where(bodys[int(len(rule[4:]) / 3) - 1] == 'Y')[0][0] == 1 else 2)] if is_connected_closed else [
        np.where((head == 'Y') | (head == 'X'))[0][0],
        np.where((bodys[int(len(rule[4:]) / 3) - 1] == 'Y') | (bodys[int(len(rule[4:]) / 3) - 1] == 'X'))[0][0]]
    # check the type of specific rule
    is_X_in_head = True if ('X' in head and 'Y' not in head) else False
    # check if the KG is AKG
    is_attribute_KG = (attribite_dic is not None and attributes is not None)
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

    # the head of the feature tables
    tablehead = []
    for body in bodys:
        tablehead += (body[1] + '_is_' + pd.Series(relations) + '_of').tolist()
        tablehead += (body[2] + '_has_' + pd.Series(relations)).tolist()
    # labels
    Y = []
    # recursive head matching pairs
    headmatching = []
    # numerator of SC
    numerator = 0
    # rule head matching
    for result in results:
        bodys = np.array(result).reshape(int(len(result) / 3), 3)
        # check if the pair match the rule head
        if is_connected_closed:
            if [bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]] in relation_dic:
                numerator += 1
                Y.append(1)
            else:
                Y.append(0)
            headmatching.append([bodys[0][index_head[0]], head[0], bodys[int(len(result) / 3) - 1][index_head[1]]])
        elif not is_connected_closed and is_X_in_head and index_head[0] == 1:
            if [bodys[int(len(result) / 3) - 1][index_head[1]], head[0], head[2]] in relation_dic:
                numerator += 1
                Y.append(1)
            else:
                Y.append(0)
            headmatching.append([bodys[int(len(result) / 3) - 1][index_head[1]], head[0], head[2]])
        elif not is_connected_closed and not is_X_in_head and index_head[0] == 2:
            if [head[1], head[0], bodys[int(len(result) / 3) - 1][index_head[1]], head[0]] in relation_dic:
                numerator += 1
                Y.append(1)
            else:
                Y.append(0)
                headmatching.append([head[1], head[0], bodys[int(len(result) / 3) - 1][index_head[1]], head[0]])
    # denominator of SC
    denominator = len(results)
    if numerator < 2 or denominator <= 3:
        return None, None, 0, 0, None, None
    # features
    features = []
    for result in results:
        bodys = np.array(result).reshape(int(len(result) / 3), 3)
        # features
        feature = []
        for body in bodys:
            feature.extend(get_attribute(body, relations, relation_dic))
        features.append(feature)
    # Extra attributes from AKG
    if is_attribute_KG:
        extra_features = []
        for result in results:
            bodys = np.array(result).reshape(int(len(result) / 3), 3)
            # extra features from AKG
            extra_feature, extra_table_head = get_extra_attribute(bodys, attributes,
                                                                  attribite_dic,
                                                                  contains_numerical_attribute)
            extra_features.append(extra_feature)
            tablehead.extend(extra_table_head)
        # transform categorical values to numerical values
        if contains_numerical_attribute == 0:
            extra_features = target_encoding(extra_features, Y)
        features = np.hstack((features, extra_features))

    # simplify attribute matrix/ remove spareness
    tablehead, dataframe = simplify_datainput(tablehead, features)
    return dataframe, Y, numerator, denominator, tablehead, headmatching


# check if the entity is a general entity in the form of "X" for example
def is_general_entity(entity):
    return len(entity) == 1

# matching mechanism that find triplet sets that satisfy rule body
def matchhelp(former, current, body, candidates, relation_dic):
    """
    :param former: previous connected entity index if exist
    :param current: current connected entity index if exist
    :param body: current body
    :param candidates: results series that match the conditions
    :param relation_dic: facts in KG
    :return: results series that match the current conditions
    """
    # target relation
    relation, entity1, entity2 = body[0], body[1], body[2]
    # print(entity1, relation, entity2)
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


def augmented_rules(i, relations, relation_dic, rules, sc_min, sc_max, head_coverage_threshold,
                    support_threshold,
                    data_set,
                    attributes=None, attribite_dic=None,
                    contains_numerical_attribute=0):
    '''
    :param i: index of rules
    :param relations: relational facts in KG
    :param relation_dic: relations in KG
    :param rules: rules from rule-based system
    :param sc_min: threshold
    :param sc_max: threshold
    :param head_coverage_threshold: threshold
    :param support_threshold: threshold
    :param data_set: data set (training set by default)
    :param attributes: attributive facts in AKG if applicable
    :param attribite_dic: attributes in AKG if applicable
    :param contains_numerical_attribute: boolean paramter that checks if the AKG contain numerical attributes
    :return: boolean index that indicate if the rule is augmented
    '''
    # extract current rule
    cur_rule = re.sub(r'\t|\n|<--|\(|\)|,', ' ', rules[i]).split()
    cur_rule = cur_rule[2:6] + cur_rule[7:]
    # rules have to meet some conditions: SC is greater than a threshold e.g.0.5)
    if not rule_pre_check(cur_rule, sc_min, sc_max):
        # print("No. %d" % i, " Original rule does not meet the pre-conditions")
        return 0
    # print("No. %d" % i, " Original rule meets the pre-conditions")
    # check if the rule is connected-closed rule
    is_connected_closed = connected_closed_rule_check(cur_rule)
    # Data preparation
    X, Y, numerator, denominator, attributehead, headmatching = match(cur_rule, is_connected_closed, relations,
                                                                      relation_dic,
                                                                      attributes,
                                                                      attribite_dic,
                                                                      contains_numerical_attribute)  # , attribute_mapping_dic)

    # DT construction requirement:
    # support is at least greater than support_threshold and the number in each class should be greater than 2
    if X is None or Y is None or numerator < support_threshold or denominator - numerator < 2 or numerator == denominator:
        return 0
    print("No. %d " % i, "connected ", is_connected_closed, numerator, denominator, np.array(Y).shape,
          np.array(X).shape)

    # Calculate the head coverage
    head_coverage = get_headcoverage(cur_rule, numerator, relation_dic)
    if head_coverage < head_coverage_threshold:
        return 0
    # resampling for balanced data set
    if np.array(X).shape[0] < np.array(X).shape[1]:
        x_sample, y_sample = SMOTE(k_neighbors=2).fit_resample(np.array(X), np.array(Y).ravel())
    else:
        x_sample, y_sample = X, Y
    """    Classifcation model with Grid Search    """
    # Decision Tree Model-Cart default=”gini”
    params = {'max_features': ['auto', 'sqrt', 'log2'],
                  'ccp_alpha': [0.1, .01, .001],
                  'max_depth': [5, 7, 9]
                  }
    model = tree.DecisionTreeClassifier(splitter='random', class_weight='balanced',criterion='gini')

    # # Decision Tree Model-ID3
    # params = {'max_features': ['auto', 'sqrt', 'log2'],
    #           'ccp_alpha': [0.1, .01, .001],
    #           'max_depth': [5, 6, 7, 8, 9]
    #           }
    # model = tree.DecisionTreeClassifier(splitter='random', class_weight='balanced', criterion='entropy')

    # # Random Forest Model
    # params = {
    #     'n_estimators': [10, 50, 100],
    #     'max_depth': [2, 4, 6],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'min_samples_leaf': [2, 5, 10],
    #     'criterion': ['gini', 'entropy']
    # }
    # model = RandomForestClassifier(class_weight="balanced")

    # # Logistic Regression
    # params = {"C": np.logspace(-3, 3, 7),
    #           "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    # model = LogisticRegression()

    # # Gradient Boosting Classifier
    # params = {'learning_rate': np.linspace(0.05, 0.25, 5), 'max_depth': [x for x in range(1, 8, 1)],
    #           'min_samples_leaf': [x for x in range(1, 5, 1)], 'n_estimators': [x for x in range(50, 100, 10)]}
    # model = GradientBoostingClassifier()

    # #AdaBoost Classifier
    # params = {'base_estimator__max_depth': [i for i in range(2, 11, 2)],
    #               'base_estimator__min_samples_leaf': [5, 10],
    #               'n_estimators': [10, 50, 250, 1000],
    #               'learning_rate': [0.01, 0.1]}
    # model = AdaBoostClassifier()

    # Cross Validation with Grid search
    grid = GridSearchCV(model, params, cv=10, scoring="average_precision")
    grid.fit(x_sample, y_sample)
    best_model = grid.best_estimator_

    # Fit the data with early-stopping
    for r in range(1, 20):
        best_model.fit(x_sample, y_sample)
        pred_y = best_model.predict(X)
        tn, fp, fn, tp = confusion_matrix(Y, pred_y).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        # new SC needs to satisfy precision & recall requirement
        if recall > 0.8 and 0.8 > precision > 0.5:
            break

    # # ! DT visualization only applicable to decision-tree models
    # text_representation = tree.export_text(best_model,feature_names=attributehead)
    # print(text_representation)
    # # Model result visualization
    # plt.figure(figsize=(15,15))  # set plot size (denoted in inches)
    # cn=['Nohead',"Head"]
    # tree.plot_tree(best_model,max_depth=3,feature_names = attributehead, class_names=cn,filled = True,fontsize=8)


    # model prediction
    pred_y = best_model.predict(X)
    tn, fp, fn, tp = confusion_matrix(Y, pred_y).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # record the SC improvement
    with open('rules/' + data_set + '/SC_improve', 'a') as f:
        text=str(numerator / denominator)+"\t"+str(tp / (tp + fp))
        f.write(text + '\r\n')

    # ignore those with no SC improvement & ignore those over-fitting to training set
    if precision > 0.8 or precision < 0.6 or recall < 0.6 or np.isnan(
            tp / (tp + fp)) or numerator / denominator > precision:
        return 0

    # records new facts
    print("Precision on X: ", str(precision), "Recall on X: ", str(recall))
    # true positive + false positive
    unq = np.array([x + 2 * y for x, y in zip(pred_y, Y)])
    res_index = np.array(np.where(unq >= 2)).tolist()[0]
    results = [[re.sub('\n', '', rules[i])] + headmatching[j] + [str(precision)] for j in res_index]
    with open('rules/' + data_set + '/augmented_rule', 'a') as f:
        if results is not None:
            for r in results:
                f.write("\t".join(r) + '\r\n')
    print("Current index: ", str(i), " is_connected: ", is_connected_closed, " Add ",
          len(results), " Records")
    return 1

# define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)


if __name__ == '__main__':
    # process hyper-parameters
    paramters = sys.argv[1:]
    data_set, sc_min, sc_max, head_coverage, support_threshold, learning_time, contains_numerical_attribute = \
        paramters[0], float(paramters[1]), float(paramters[2]), float(paramters[3]), int(paramters[4]), str(
            paramters[5]), int(paramters[6])
    print("Current hyper-parameters: ", data_set, sc_min, sc_max, head_coverage, support_threshold, learning_time,
          contains_numerical_attribute)
    # read rules
    rules = rule_extract("rules/" + data_set + "/alpha-" + learning_time)
    # reading predicates
    relation_dic = []
    relation_dic.extend(get_relation_dic("data 3/" + data_set + "/train.txt"))
    relation_dic.extend(get_relation_dic("data 3/" + data_set + "/valid.txt"))
    # reading relations
    relations = set()
    for pair in relation_dic:
        relations.add(pair[1])
    relations = list(relations)
    # read entities
    entities = entity_inspection(relation_dic)

    # Attributes
    attribite_dic = get_attribute_dic("data 3/" + data_set, contains_numerical_attribute)
    attributes = get_attributes("data 3/" + data_set + "/attribute2id.txt")
    # attribute_mapping_dic = categorical_to_discrete("data 3/" + data_set + "/attribute_val.txt")
    print("The number of facts: ", len(relation_dic))
    print("The number of relations: ", len(relations))
    print("The number of entities: ", len(entities))
    print("The number of rules: ", len(rules))
    if attribite_dic is not None and attributes is not None:
        print("The number of attribute dic: ", len(attribite_dic))
        print("The number of attribute type: ", len(attributes))

    # Asynchronous parallel computing
    T1 = time.time()
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # # Step 2: Use loop to parallelize
    for i in range(0, len(rules)):
        pool.apply_async(augmented_rules,
                         args=(
                             i, relations, relation_dic, rules, sc_min, sc_max, head_coverage,
                             support_threshold, data_set,
                             attributes, attribite_dic, contains_numerical_attribute),
                         callback=collect_result)
    # Step 3: Close Pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    T2 = time.time()
    print('The Augmentation program runs about :%s s' % (T2 - T1), "Got %s augmented rules" % (sum(results)))