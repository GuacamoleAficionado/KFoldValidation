"""
Author        : Zach Seiss
Email         : zseiss2997@gmail.com
Date Created  : May 31, 2022
Last Update   : June 5, 2022
"""

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
        sorted_index = sorted(data_frame[target].value_counts().index)
        sorted_series = data_frame[target].value_counts().reindex(sorted_index)
        return np.expand_dims((sorted_series / data_frame[target].count()), axis=1)

    '''
    Otherwise we need to do some extra processing. 
    '''
    all_variables = list(givens) + [target]
    df = data_frame[all_variables]
    grouped = df.groupby(list(givens))
    val_counts = grouped[target].value_counts()
    counts = grouped[target].count()

    '''
    We need to collect the Series objects that make up the DataFrame we're
    using to make the CPT.  This is overly complicated but it is due to
    an obnoxious behaviour of pandas which will be elaborated shortly.
    '''

    series = [sorted(df.iloc[:, i].unique()) for i in range(len(df.columns))]

    '''
    We need the Series objects because pandas exhibits a particular behaviour
    in which the rows of a groupby object that are not 'instantiated'
    simply get dropped.  This behaviour is absolutely not
    what we want and there is no simple argument that we can pass to the the 
    groupby method to change it.  The solution we came up with is to reindex
    each groupby object with the cartesian product of all the series' which
    make up that particular DataFrame object.
    '''
    val_counts_multi_index = pd.MultiIndex.from_product(series)
    val_counts = val_counts.reindex(val_counts_multi_index, fill_value=0)

    '''
    For some reason if you use a multi-index with only a single level to
    reindex a series the values all get switched to Nan so to counteract 
    this we check the levels in the multi-index and change it to a standard
    list if there is only a single level. 
    '''
    counts_multi_index = pd.MultiIndex.from_product(series[:-1])
    counts_multi_index = counts_multi_index if counts_multi_index.nlevels > 1 \
        else counts_multi_index.levels[0]
    counts = counts.reindex(counts_multi_index, fill_value=0)

    '''
    After reindexing, for some reason, we need to modify 'counts' manually
    in order to make the division of val_counts by counts work properly.
    We might look into why this is if we get time. 
    '''
    target_variable_cardinality = len(df[target].unique())
    arr = np.array(val_counts / np.repeat(counts.values, target_variable_cardinality))

    '''
    'unprocessed_cpd' is the conditional probability distribution as a flat list.  We need to process
    it to get it into the proper format to input into pgmpy's 'TabularCPD()' constructor.
    '''
    unprocessed_cpd = np.nan_to_num(arr)

    '''
    total_states_of_givens is the product of the number of states of all the 'given' variables in the space
    that the table is concerned with.
    '''
    variable_states = [len(df[elem].unique()) for elem in givens]
    total_states_of_givens = math.prod(variable_states)

    #  'cpt' is the tentative return value.
    cpt = np.array(np.split(unprocessed_cpd, total_states_of_givens)).T

    '''  Now we check to see if any of the columns of the CPT are invalid.  
         This particular solution only works when the target variable is binary.
         That's fine for our project but to get more general would require 
         some thought.'''
         
    '''  It might seem here that rather than use a logical_and operation we
         should simply check to see that cpt[0] + cpt[1] == 1, that is, the
         columns of the CPT sum to 1.  However, we specifically want to look
         for the case where the sum of the column is zero.  If there is a 
         case where 0 < sum(column) < 1 then that would be indicative of a
         fundamentally different kind of error and we want the interpreter
         to raise an Exception to notify us
    '''
    problem_cols = (np.logical_and(cpt[0] == cpt[1], cpt[0] == 0))
    if np.any(problem_cols):
        problem_index = [i for i in range(problem_cols.size) if problem_cols[i]]
        num_problems = len(problem_index)
        num_cols = np.shape(cpt)[1]
        avg_zeroth_row = np.sum(cpt[0]) / (num_cols - num_problems)
        avg_first_row = np.sum(cpt[1]) / (num_cols - num_problems)
        cpt[0][problem_index] = avg_zeroth_row
        cpt[1][problem_index] = avg_first_row

    return cpt


#  Example ___________________________________________________________________

# data = pd.read_csv('ACS_modified.csv')
# my_cpd = make_cpd(data, 'LikesProduct', 'Product', 'DenominationalGroup', 'StateCleaned')
# print(np.shape(my_cpd))
