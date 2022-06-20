"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    May 25, 2022
Last Update  :    June 5, 2022
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime
from time import time

from bayes_net_model import make_bn
from optimized_query import fast_query
from get_client_spreadsheet import return_client_csv

# we will define variables begin and end to keep track of program execution time
begin = time()

random.seed(0)
DROP_CUTOFF = 20
df = pd.read_csv('ACST_Cust_Data.csv')
df = df.loc[df['MissingValues'] <= DROP_CUTOFF].reset_index()
K = 10
NUM_ROWS = len(df)
TARGET_VARIABLE = 'Satisfied'

index = list(range(len(df)))
random_sample = df.iloc[np.array(random.sample(index, NUM_ROWS))]
sample_index = random_sample.index

'''
We let n = length of index mod k.  That is, if we split the random sample into k equal sized groups,
there will be n < k remaining unassigned elements in the sample.  So we remove the last n elements from 
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
    bn = make_bn(training_groups[i], [('DenominationalGroup', 'Satisfied'),
                                      ('Party', 'Satisfied'),
                                      ('TWA_grouped', 'Satisfied'),
                                      ('TWA_grouped', 'CongregantUsers_grouped'),
                                      ('Region', 'Satisfied'),
                                      ('Product', 'Satisfied'),
                                      ('Satisfied', 'YearsOwned'),
                                      ('Satisfied', 'UsingAccounting'),
                                      ('Satisfied', 'UsingBackGroundChecks'),
                                      ('Satisfied', 'UsingCheckin'),
                                      ('Satisfied', 'UsingMobileCheckin'),
                                      ('Satisfied', 'UsingMinistrySmart'),
                                      ('Satisfied', 'UsingRefreshWebsites'),
                                      ('Satisfied', 'UsingOnlineGiving'),
                                      ('Satisfied', 'UsingPathways'),
                                      ('Satisfied', 'UsingMissionInsite'),
                                      ('Satisfied', 'CongregantUsers_grouped'),
                                      ])

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

# num_queries=0
# method_used='uniform'
# external_errors=0
# file_name=''
fq, num_queries, method_used, external_errors = fast_query(bayesian_networks,
                                                           test_group_indexes,
                                                           environment_variables,
                                                           df,
                                                           TARGET_VARIABLE)

validations = []
high_risk_group = []
moderate_risk_group = []
false_negatives = []
error_count = 0
for i in range(K):
    validation = []
    false_negative_lst = []
    for j in range(test_group_sizes[i]):
        #  'state_instantiation' is a Series from which we can obtain the instantiated state variables.
        state_instantiation = test_groups[i].iloc[j][environment_variables]

        #  'prediction' is the max likelihood state of the target variable given the states of the other variables.
        client_ID = test_groups[i].iloc[j]['ID']
        prediction = fq[i].loc[tuple(state_instantiation.values)]['0_y']
        #  'actual_target_value' is the true value of the state variable we are trying to predict.
        actual_target_value = test_groups[i].iloc[j][TARGET_VARIABLE]
        try:
            if (prediction < .60) and (prediction > .5) and actual_target_value:
                moderate_risk_group.append((client_ID, prediction))
            validation.append((prediction > .5) == actual_target_value)
            if not (round(prediction)) and actual_target_value:
                # client_ID - ID in data set of client about whom we are making a prediction.
                false_negative_lst.append(client_ID)
                high_risk_group.append((client_ID, prediction))
        except ValueError as e:
            error_count += 1
            print(e)
        except TypeError as e:
            error_count += 1
            print(prediction)
            print(e)
    false_negatives.append(false_negative_lst)
    validations.append(np.array(validation))
#  rc_sizes is the number of false negatives in each testing group which we use in an error computation later.
rc_sizes = np.array([len(lst) for lst in false_negatives])
# data_on_risky_clients = df.loc[df['ID'].isin(risky_clients)]


"""                             ERROR CALCULATION                           """
num_correct_predictions = np.array([np.sum(validation) for validation in validations])
group_prediction_accuracies = num_correct_predictions / (test_group_sizes - error_count)
group_prediction_accuracies_fn = num_correct_predictions / (test_group_sizes - rc_sizes - error_count)
mean_fn = np.mean(group_prediction_accuracies_fn)
std_fn = np.std(group_prediction_accuracies_fn)
mean = np.mean(group_prediction_accuracies)
std = np.std(group_prediction_accuracies)

"""                             REPORT PRINTING                             """
date_stamp = datetime.now()
end = time()
bn = bayesian_networks[0]
file_name = return_client_csv(high_risk_lst=high_risk_group,
                              moderate_risk_lst=moderate_risk_group,
                              data_frame=df)
report = f'###################################################      {file_name}      {date_stamp}      >{DROP_CUTOFF} MissingValues dropped!!!   ##################################################\n\n' \
         f'Method Used : {method_used}\n' \
         f'Prediction Accuracy : {round(mean, 5)}\n' \
         f'Standard Deviation : {round(std, 5)}\n' \
         f'Accuracy without "false negatives" : {round(mean_fn, 5)}\n' \
         f'Standard Deviation without "false negatives" : {round(std_fn, 5)}\n' \
         f'Execution Time : {round(((end - begin) / 60), 2)} minutes\n' \
         f'The network was queried {num_queries} times.  FastQuery saved {len(df) - num_queries} redundant queries.\n' \
         f'Error count : {error_count + external_errors}\n' \
         f'Nodes : {bn.nodes}\n' \
         f'Edges : {bn.edges}\n' \
         f'In Degree : {bn.in_degree}\n' \
         f'Out Degree : {bn.out_degree}\n' \
         f'States : {bn.states}\n\n\n\n'
print(report)
with open('BN_testing_new_query_evidence_style.txt', 'a') as file:
    file.write('\n\n' + report)
with open(f'Client_Spreadsheets/{file_name}/{file_name}.txt', 'w+') as file:
    file.write(report)
