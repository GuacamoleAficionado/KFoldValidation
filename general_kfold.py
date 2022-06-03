"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    May 25, 2022
Last Update  :    June 1, 2022
"""

import numpy as np
import pandas as pd
import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.ExactInference import VariableElimination
from cpdmaker import make_cpd

random.seed(0)
df = pd.read_csv('ACS_modified.csv')
K = 10
NUM_ROWS = len(df)
TARGET_VARIABLE = 'Deactivated'


def state_mapping(data_frame, variable):
    """__________________________________________________________________________________________
        Given an enivronment variable this function returns a dictionary mapping every state
        of the variable to the integer that corresponds to its index in the conditional
        probability table.
       _________________________________________________________________________________________"""
    return dict([(b, a) for a, b in enumerate(sorted(data_frame[variable].unique()))])


def environment_map(data_frame, universe):
    """__________________________________________________________________________________________
         User should pass in the list of all environment variables as 'universe'.  The returned 
         object is a nested dictionary mapping each environment variable to the dictionary which  
         maps its respective states to their indexes in the appropriate CPT.
       __________________________________________________________________________________________"""
    return {variable: state_mapping(df, variable) for variable in universe}


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
#  We train each BN inside a for loop and then store each in a list called 'bayesian_networks'
bayesian_networks = []
for i in range(K):
    bn = BayesianNetwork([('DenominationalGroup', 'Deactivated'), ('Deactivated', 'CongregantUsers')])

    denominational_group = TabularCPD('DenominationalGroup', 18,
                                      values=make_cpd(training_groups[i], 'DenominationalGroup'))

    deactivated_cpd = TabularCPD('Deactivated', 2,
                                 values=make_cpd(training_groups[i], 'Deactivated', 'DenominationalGroup'),
                                 evidence=['DenominationalGroup'], evidence_card=[18])

    congregant_users_cpd = TabularCPD('CongregantUsers', 4,
                                      values=make_cpd(training_groups[i], 'CongregantUsers', 'Deactivated'),
                                      evidence=['Deactivated'], evidence_card=[2])

    bn.add_cpds(denominational_group, deactivated_cpd, congregant_users_cpd)
    bayesian_networks.append(bn)

'''
Now we have to create a VariableElimination object from each BayesianNetwork object in order
to do inference (we need to run queries).
'''

test_groups = [df.iloc[test_group_indexes[i]] for i in range(K)]
test_group_sizes = np.array([elem.size for elem in test_group_indexes])
train_group_sizes = np.array([elem.size for elem in train_group_indexes])

'''
Importantly, we can not assume that every item in our testing group has been accounted for in the
training group e.g. 'AME Zion' might be in a testing group when there were no instances of this 
denomination in the training group so we cannot make a prediction in regard to it (probably?).
'''

inferences = [VariableElimination(bn) for bn in bayesian_networks]
environment_variables = [variable for variable in bayesian_networks[0]]
environment_variables.remove(TARGET_VARIABLE)
env_map = environment_map(df, environment_variables)
validations = []
for i in range(K):
    validation = []
    for j in range(test_group_sizes[i]):
        states = test_groups[i].iloc[j][environment_variables]
        true_value_target = test_groups[i].iloc[j][TARGET_VARIABLE]
        inference = inferences[i].query([TARGET_VARIABLE],
                                        {v: env_map[v][s] for v, s in zip(environment_variables, states)},
                                        show_progress=False)
        validation.append((inference.values[0] < .5) == true_value_target)
    validations.append(np.array(validation))

group_prediction_accuracies = np.array([np.sum(validation) for validation in validations]) / test_group_sizes

total_accuracy = np.sum(group_prediction_accuracies) / K
print(f'Prediction Accuracy : {total_accuracy}')
