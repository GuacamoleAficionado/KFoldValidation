import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.ExactInference import VariableElimination
import cpdmaker

'''  Create the BayesianNetwork object  '''
bn = BayesianNetwork([('DenominationalGroup', 'Deactivated'), ('Deactivated', 'CongregantUsers')])

'''  Create the conditional probability tables  '''
df = pd.read_csv('ACS_modified.csv')

denominational_group = TabularCPD('DenominationalGroup', 18,
                                  values=np.expand_dims(  # Need to make cptmaker able to calculate priors!!!!!!!!!
                                      (df.DenominationalGroup.value_counts() / df.DenominationalGroup.count()), axis=1))

deactivated_cpd = TabularCPD('Deactivated', 2, values=cpdmaker.make_cpd(df, 'Deactivated', 'DenominationalGroup'),
                             evidence=['DenominationalGroup'], evidence_card=[18])

congregant_users_cpd = TabularCPD('CongregantUsers', 4, values=cpdmaker.make_cpd(df, 'CongregantUsers', 'Deactivated'),
                                  evidence=['Deactivated'], evidence_card=[2])

'''  Add the CPDs to the network  '''
bn.add_cpds(denominational_group, deactivated_cpd, congregant_users_cpd)

'''  Create the VariableElimination() object to query  '''
inference = VariableElimination(bn)

# print(inference.query(variables=['Deactivated'], evidence={'DenominationalGroup' : 3, 'CongregantUsers' : 1}))
