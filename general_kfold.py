"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    May 25, 2022
Last Update  :    June 5, 2022
"""

import numpy as np
import pandas as pd
import random
from time import time
from bayes_net_model import make_bn
from optimized_query import fast_query

# we will define variables begin and end to keep track of program execution time
begin = time()

random.seed(0)
df = pd.read_csv('ACS_modified.csv')
K = 10
NUM_ROWS = len(df)
TARGET_VARIABLE = 'LikesProduct'

index = list(range(len(df)))
random_sample = df.iloc[np.array(random.sample(index, NUM_ROWS))]
sample_index = random_sample.index

'''
We let n = length of index mod k.  That is, if we split the random sample into k equal sized groups,
there will be n remaining unassigned elements in the sample.  So we remove the last n elements from 
index, split index into k equal sized groups then assign the remaining n elements to the first n of the 
k groups.
'''
n = NUM_ROWS % K
remaining_indices = sample_index[NUM_ROWS - n:]
mod_sample_index = sample_index[: NUM_ROWS - n]
test_group_indexes = np.split(np.array(mod_sample_index), K)
for i in range(n):
    test_group_indexes[i] = np.append(test_group_indexes[i], (remaining_indices[i]))

'''
We get a list of the training group indexes by removing the 
indexes associated with the ith testing group from the list of indices of
the full data set
'''
# train_group_indexes is an array of length k which contains the index for each training group
train_group_indexes = [sample_index.drop(test_group_indexes[i]) for i in range(K)]
training_groups = [df.iloc[train_group_indexes[i]] for i in range(K)]

''' 
for each training group we have to train a new BN.  Then we will query that BN for each member
of the associated testing group and compare its max likelihood prediction against the true value.
'''

'''  AS CURRENTLY IMPLEMENTED, THIS PROGRAM WILL FAIL FOR LESS THAN 3 NODE BNs!!!!  
'''
bayesian_networks = []
for i in range(K):
    bn = make_bn(training_groups[i], [('Product', 'LikesProduct'),
                                      ('StateCleaned', 'LikesProduct'),
                                      ('DenominationalGroup', 'LikesProduct'),
                                      # ('LikesProduct', 'CongregantUsers')])
                                      ('LikesProduct', 'UsingOnlineGiving'),
                                      ('LikesProduct', 'Timeline'),
                                      ('LikesProduct', 'UsingPathways'),
                                      ('LikesProduct', 'UsingMissionInsite')])
    # bn.check_model()
    bayesian_networks.append(bn)

'''
Now we have to create a VariableElimination object from each BayesianNetwork object in order
to do inference (we need to run queries).
'''

test_groups = [df.iloc[test_group_indexes[i]] for i in range(K)]
test_group_sizes = np.array([elem.size for elem in test_group_indexes])
train_group_sizes = np.array([elem.size for elem in train_group_indexes])

environment_variables = [variable for variable in bayesian_networks[0]]
environment_variables.remove(TARGET_VARIABLE)

''' 
The function fast_query will query all of the bayesian networks with the whole environment
map and map the queries to their respective outputs, reducing computation time by 
eliminating repeat calculations.
'''
fq = fast_query(bayesian_networks,
                environment_variables,
                df,
                TARGET_VARIABLE)
validations = []
risky_clients = np.array([])
for i in range(K):
    validation = []
    false_positives = np.array([])
    for j in range(test_group_sizes[i]):
        #  'state_instantiation' is a Series from which we can obtain the instantiated state variables.
        state_instantiation = test_groups[i].iloc[j][environment_variables]
        #  'prediction' is the max likelihood state of the target variable given the states of the other variables.
        prediction = fq[i].loc[tuple(state_instantiation.values)]['0_y']
        # 'client_ID' is the ID in the dataset that refers to the client about whom we are currently making a
        # prediction.
        client_ID = test_groups[i].iloc[j]['ID']
        #  'actual_target_value' is the true value of the state variable we are trying to predict.
        actual_target_value = test_groups[i].iloc[j][TARGET_VARIABLE]
        validation.append(prediction == actual_target_value)
        if not prediction and actual_target_value:
            false_positives = np.append(false_positives, client_ID)
    risky_clients = np.append(risky_clients, false_positives)
    validations.append(np.array(validation))
risky_clients = risky_clients.flatten()
# data_on_risky_clients = df.loc[df['ID'].isin(risky_clients)]
group_prediction_accuracies = np.array([np.sum(validation) for validation in validations]) / test_group_sizes

############################################  REPORT PRINTING  ###############################################
std_dev = np.std(group_prediction_accuracies)
total_accuracy = np.sum(group_prediction_accuracies) / K
end = time()
bn = bayesian_networks[0]
report = f'Prediction Accuracy : {round(total_accuracy, 5)}\n' \
         f'Standard Deviation : {round(std_dev, 5)}\n' \
         f'Execution Time : {round(((end - begin) / 60), 2)} minutes\n' \
         f'Nodes : {bn.nodes}\n' \
         f'Edges : {bn.edges}\n' \
         f'In Degree : {bn.in_degree}\n' \
         f'Out Degree : {bn.out_degree}\n' \
         f'States : {bn.states}'
print(report)
with open('/home/zach/Desktop/Some sample BN testing', 'a') as file:
   file.write('\n\n' + report)
##############################################################################################################
