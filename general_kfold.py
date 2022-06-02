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


df = pd.read_csv('ACS_modified.csv')

#environment_variables = [elem.variable for elem in tv.bn.get_cpds()]


'''  Given a variable this function returns a dictionary mapping every state
     of the variable to the integer that corresponds to its index in the CPT.
'''


def state_mapping(data_frame, variable):
    return dict([(b, a) for a, b in enumerate(sorted(data_frame[variable].unique()))])


'''  Pass in the list of all environment variables as 'universe'.  The returned object
     is a nested dictionary mapping each environment variable which maps its respective
     state variables to their indexes in the appropriate CPT.
'''


def environment_map(universe):
    return {variable: state_mapping(df, variable) for variable in universe}


SIZE_OF_DATA = len(df)
K = 10
index = list(range(len(df)))
random_sample = df.iloc[np.array(random.sample(index, SIZE_OF_DATA))]
sample_index = random_sample.index

'''
We let n = length of index mod k.  That is, if we split the random sample into k equal sized groups,
there will be n remaining unassigned elements in the sample.  So we remove the last n elements from 
index, split index into k equal sized groups then assign the remaining n elements to the first n of the 
k groups.
'''
n = SIZE_OF_DATA % K
remaining_indices = sample_index[SIZE_OF_DATA - n:]
mod_sample_index = sample_index[: SIZE_OF_DATA - n]
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

    #  We need to do this but generally
    denominational_group = TabularCPD('DenominationalGroup', 18,
                                      values=make_cpd(training_groups[i], 'DenominationalGroup'))

    deactivated_cpd = TabularCPD('Deactivated', 2, values=make_cpd(training_groups[i], 'Deactivated', 'DenominationalGroup'),
                                 evidence=['DenominationalGroup'], evidence_card=[18])

    congregant_users_cpd = TabularCPD('CongregantUsers', 4,
                                      values=make_cpd(training_groups[i], 'CongregantUsers', 'Deactivated'),
                                      evidence=['Deactivated'], evidence_card=[2])
    bayesian_networks.append(bn)
print(bayesian_networks)

def max_likelihood(x):
    """  We input 'x', the probability of a boolean event, and the output
         is whether the event is more likely than not to occur.
    """
    return x > 0.5


def training_predictions(assign, training):
    """
                    FUNCTION INPUTS training_predictions()
    ------------------------------------------------------------------------------

      assign   - a function which maps a training value to a prediction

      training - a DataFrame object whose index is a 'condition' and whose
                 values are the probability of the event of interest given
                 the corresponding condition in index.
    ------------------------------------------------------------------------------
    """
    return [dict(zip(training[i].index, list(map(assign, training[i].values)))) for i in range(K)]


'''
The ith element of training_predictions is a dictionary corresponding to the ith training group which 
maps the name of each denomination to our prediction as to whether the church will deactivate or 
not given the denomination.
'''
predictions = training_predictions(max_likelihood, t_g_processed)
test_groups = [df.iloc[test_group_indexes[i]] for i in range(K)]

'''
For each ith test group, we need to pass the index (DenominationalGroup) to the ith dictionary in 
'predictions' and check whether predictions maps to the same boolean value as actually corresponds
with this entry in the the table for 'Deactivated'

Importantly, we can not assume that every item in our testing group has been accounted for in the
training group e.g. 'AME Zion' might be in a testing group when there were no instances of this 
denomination in the training group so we cannot make a prediction in regard to it.  Therefore, 
one of the first things we need to do is iterate through every ith testing group and drop all 
rows which do not have a corresponding key in the ith dictionary of 'predictions.'
'''

validations = []
for i in range(K):
    validation = []
    for j in range(len(test_groups[i])):
        validation.append(predictions[i][test_groups[i].values[j][0]] == test_groups[i].iloc[j]['bool_data'])
    validations.append(validation)
# test_group_sizes is an array of length k in which the ith element is the size of the ith test-group

test_group_sizes = np.array([elem.size for elem in test_group_indexes])
train_group_sizes = np.array([elem.size for elem in train_group_indexes])

group_prediction_accuracies = np.array([np.sum(validation) for validation in validations]) / test_group_sizes
print(np.sum(group_prediction_accuracies) / K)
