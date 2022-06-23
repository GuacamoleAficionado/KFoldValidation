"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    June 22, 2022
Last Update  :    June 22, 2022
"""

import pandas as pd
import random
from datetime import datetime
from time import time

from bayes_net_model import make_bn
from optimized_query import fast_query
from get_client_spreadsheet import return_client_csv

# we will define variables begin and end to keep track of program execution time
begin = time()

random.seed(100)
DROP_CUTOFF = 20
df = pd.read_csv('ACST_Cust_Data.csv')
NUM_ROWS = len(df)
TARGET_VARIABLE = 'Satisfied'
index = list(range(len(df)))

'''  AS CURRENTLY IMPLEMENTED, THIS PROGRAM WILL FAIL FOR LESS THAN 3 NODE BNs!!!!  
'''
bayesian_network = make_bn(df, [('DenominationalGroup', 'Satisfied'),
                                ('Satisfied', 'CongregantUsers_grouped'),
                                ('Satisfied', 'YearsOwned'),
                                ('Satisfied', 'UsingOpenInvitationModel'),
                                ('Satisfied', 'UsingMissionInsite'),
                                ('Satisfied', 'UsingAccounting'),
                                ('Satisfied', 'UsingOnlineGiving'),
                                ('Satisfied', 'UsingBackGroundChecks'),
                                ('Satisfied', 'UsingPathways'),
                                ('Satisfied', 'UsingCheckin'),
                                ('Satisfied', 'UsingMobileCheckin'),
                                ('Satisfied', 'UsingMinistrySmart'),
                                ('Satisfied', 'UsingRefreshWebsites'),
                                ('Party', 'Satisfied'),
                                ('Product', 'Satisfied'),
                                ('MissingValues', 'Satisfied'),
                                ('Region', 'Satisfied'),
                                ('TWA_grouped', 'Satisfied'),
                                ('TWA_grouped', 'CongregantUsers_grouped')])

environment_variables = [variable for variable in bayesian_network]
environment_variables.remove(TARGET_VARIABLE)

''' 
The function fast_query will query all of the bayesian networks with the whole environment
map and map the queries to their respective outputs, reducing computation time by 
eliminating repeat calculations.
'''

fq, num_queries, method_used, external_errors = fast_query([bayesian_network],
                                                           index,
                                                           environment_variables,
                                                           df,
                                                           TARGET_VARIABLE)

high_risk_group = []
moderate_risk_group = []
for i in range(len(df)):
    #  'state_instantiation' is a Series from which we can obtain the instantiated state variables.
    state_instantiation = df.iloc[i][environment_variables]

    #  'prediction' is the max likelihood state of the target variable given the states of the other variables.
    client_ID = df.iloc[i]['ID']
    prediction = fq[0].loc[tuple(state_instantiation.values)]['0_y']
    #  'actual_target_value' is the true value of the state variable we are trying to predict.
    actual_target_value = df.iloc[i][TARGET_VARIABLE]
    try:
        if (prediction < .60) and (prediction > .5) and actual_target_value:
            moderate_risk_group.append((client_ID, prediction))
        elif not (round(prediction)) and actual_target_value:
            # client_ID - ID in data set of client about whom we are making a prediction.
            high_risk_group.append((client_ID, prediction))
    except (ValueError, TypeError) as e:
        print(e)

"""                             REPORT PRINTING                             """
date_stamp = datetime.now()
end = time()
file_name = return_client_csv(high_risk_lst=high_risk_group,
                              moderate_risk_lst=moderate_risk_group,
                              data_frame=df)
report = f'###################################################      {file_name}      {date_stamp}      >{DROP_CUTOFF} MissingValues dropped!!!   ##################################################\n\n' \
         f'Method Used : {method_used}\n' \
         f'Execution Time : {round(((end - begin) / 60), 2)} minutes\n' \
         f'The network was queried {num_queries} times.  FastQuery saved {len(df) - num_queries} redundant queries.\n' \
         f'Nodes : {bayesian_network.nodes}\n' \
         f'Edges : {bayesian_network.edges}\n' \
         f'In Degree : {bayesian_network.in_degree}\n' \
         f'Out Degree : {bayesian_network.out_degree}\n'
print(report)
with open('risk_groups.txt', 'a+') as file:
    file.write('\n\n' + report)
with open(f'risk_spreadsheets/{file_name}/{file_name}.txt', 'w+') as file:
    file.write(report)
