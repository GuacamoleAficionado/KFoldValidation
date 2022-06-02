"""
Author        : Zach Seiss
Email         : zseiss2997@gmail.com
Date Created  : May 31, 2022
Last Update   : June 1, 2022
"""

import numpy as np
import pandas as pd
from functools import reduce

'''  This function should take in a DataFrame object, a target 
     variable and a list of given variables.  The user may need to
     modify the dataframe before using this function.  For instance,
     if the target variable needs to be boolean but the column
     representing the target variable in the DataFrame is not
     currently boolean, the user will need to operate on the 
     column to make its values boolean before inputting here.
'''


def make_cpd(data_frame, target, *givens):
    """
                            FUNCTION INPUTS
    --------------------------------------------------------------------------------------
        data_frame   -   A pandas DataFrame object.

        target       -   A string literal containing the name of the column 
                         in 'data_frame' which represents the 'target' variable.
                         For clarity, in the conditional probability table 
                         which will be output, the format will be, "Probability 
                         of 'target' given 'givens'"

        givens       -   A variable sized input of the given variables.

        The givens MUST be input into this function in the same order that they
        are input into the 'evidence' variable in TabularCPD() or the results
        WILL NOT BE CORRECT!!! 
    --------------------------------------------------------------------------------------
   
   """
    if givens != ():

        all_variables = list(givens) + [target]
        df = data_frame[all_variables]
        grouped = df.groupby(list(givens))
        vc = grouped[target].value_counts().unstack(fill_value=0).stack()  # vc is abbreviation for 'value_counts'
        c = grouped[target].count()  # c is short for 'count'
        # The line below returns the total number of states for each 'given' variable.  We need these numbers
        # in order to calculate the correct shape for the array that we want to output.
        variable_states = [len(df[elem].unique()) for elem in givens]
        # total_states_of_givens is the product of the number of states of all the 'given' variables in the space
        # that the table is concerned with.
        total_states_of_givens = reduce(lambda x, y: x * y, variable_states)
        # unprocessed_cpd is the conditional probability distribution as a flat list.  We need to process
        # it to get it into the proper format to input into pgmpy's 'TabularCPD()' constructor.
        unprocessed_cpd = (vc / c).values
        # for num_states in variable_states:
        return np.array(np.split(unprocessed_cpd, total_states_of_givens)).T

    else:
        return np.expand_dims((data_frame[target].value_counts() / data_frame[target].count()), axis=1)

#  Example ___________________________________________________________________

#df = pd.read_csv('ACS_modified.csv')
#my_cpd = make_cpd(df, 'DenominationalGroup')
#print(my_cpd)
