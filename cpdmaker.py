"""
Author        : Zach Seiss
Email         : zseiss2997@gmail.com
Date Created  : May 31, 2022
Last Update   : June 18, 2022
"""

from functools import reduce
import numpy as np
import pandas as pd
import math

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
    '''
    If 'givens' is empty, then we want the table for a prior.
    '''
    if givens == ():
        '''  The convention we have chosen is to sort CPDTs lexicographically.  
             Without sorting and reindexing the Series object that represents the 
             the states of the environment variable, we will get a sort by value
             counts and this will cause miscalculation further down the line.'''

        sorted_index = sorted(data_frame[target].unique())
        sorted_series = data_frame[target].value_counts(normalize=True).reindex(sorted_index)
        return np.expand_dims(sorted_series, axis=1)

    '''
    Otherwise we need to do some extra processing. 
    '''
    all_variables = list(givens) + [target]
    df = data_frame[all_variables]
    grouped = df.groupby(list(givens))
    val_counts = grouped[target].value_counts(normalize=True)

    '''
    We need to collect the Series objects that make up the DataFrame we're
    using to make the CPT.  This is overly complicated but it is due to
    an obnoxious behaviour of pandas which will be elaborated shortly.
    '''
    # debugging = [df.iloc[:, i].unique() for i in range(len(df.columns))]

    series_of_df = [sorted(df.iloc[:, i].unique()) for i in range(len(df.columns))]

    '''
    We achieve two things by reindexing (below)
    '''
    multi_index = pd.MultiIndex.from_product(series_of_df)
    val_counts = val_counts.reindex(multi_index, fill_value=0)

    '''
    'unprocessed_cpd' is the conditional probability distribution as a flat list.  We need to process
    it to get it into the proper format to input into pgmpy's 'TabularCPD()' constructor.
    '''
    unprocessed_cpd = val_counts.to_numpy()

    '''
    total_states_of_givens is the product of the number of states of all the 'given' variables in the space
    that the table is concerned with.
    '''
    variable_states = [len(df[elem].unique()) for elem in givens]
    total_states_of_givens = math.prod(variable_states)

    #  'cpt' is the tentative return value.
    cpt = np.array(np.split(unprocessed_cpd, total_states_of_givens)).T

    '''  Now we check to see if any of the columns of the CPT are invalid.  
         It might seem here that we
         should check to see that the
         columns of the CPT all sum to 1.  However, we specifically want to look
         for the case where the sum of the column is zero.  If there is a 
         case where 0 < sum(column) < 1 then that would be indicative of a
         fundamentally different kind of error and we want the interpreter
         to raise an Exception to notify us
    '''
    num_rows = len(cpt)
    problem_cols = reduce(lambda x, y: x + y, [cpt[i] for i in range(num_rows)]) == 0
    if np.any(problem_cols):
        problem_index = [i for i in range(problem_cols.size) if problem_cols[i]]
        num_problems = len(problem_index)
        num_cols = np.shape(cpt)[1]
        row_avgs = [np.sum(cpt[i] / (num_cols - num_problems))
                    for i in range(num_rows)]
        for i in range(num_rows):
            cpt[i][problem_index] = row_avgs[i]
    return cpt


#  Example ___________________________________________________________________

# data = pd.read_csv('ACST_Cust_Data.csv')
# my_cpd = make_cpd(data, 'TWA_grouped')
# print(my_cpd)
