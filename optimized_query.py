"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    June 4, 2022
Last Update  :    June 5, 2022
"""
import pandas as pd
from pgmpy.inference.ExactInference import VariableElimination


def environment_map(data_frame, universe):
    """                                 FUNCTION environment_map
       __________________________________________________________________________________________
         User should pass in the list of all environment variables as 'universe'.  The returned
         object is a nested dictionary mapping each environment variable to the dictionary which
         maps its respective states to their indexes in the appropriate CPT.

         data_frame         -   A DataFrame object which contains the environment variables
                                that we are interested in.
         universe           -   An iterable containing all the names (string format) of the
                                environment variables.  Should be a subset of data_frame.columns
       __________________________________________________________________________________________"""
    return {variable: state_mapping(data_frame[variable].unique()) for variable in universe}


def state_mapping(state_space):
    """                                 FUNCTION state_mapping
        __________________________________________________________________________________________
        Given an environment variable this function returns a dictionary mapping every state
        of the variable to the integer that corresponds to its index in the conditional
        probability table.

        state_space         -   The iterable containing all possible states of an environment
                                variable.
       _________________________________________________________________________________________"""
    return dict([(b, a) for a, b in enumerate(sorted(state_space))])


def fast_query(bns: list, environment_variables: list, data_frame: pd.DataFrame, target: str):
    inferences = [VariableElimination(bn) for bn in bns]
    env_map = environment_map(data_frame, environment_variables)
    last_var = environment_variables.pop()
    multi_index = data_frame.groupby(environment_variables)[last_var].value_counts().index
    environment_variables.append(last_var)
    quick_lookup_tables = []
    for i in range(len(inferences)):
        #  You have to use .copy() here inside the loop to get a shallow copy.
        mi_copy = multi_index.copy()
        query_evidence = pd.DataFrame(mi_copy)
        for j in range(query_evidence.size):
            my_dict = {v: env_map[v][s] for v, s in zip(environment_variables, query_evidence.loc[j][0])}
            inference = inferences[0].query([target], my_dict, show_progress=False)
            query_evidence.loc[j][0] = inference.values[0] < .5
        mymap = pd.DataFrame(range(len(multi_index)), index=mi_copy)
        quick_lookup = pd.merge(mymap, query_evidence, left_on=mymap.columns[0], right_index=True)
        quick_lookup_tables.append(quick_lookup)
    return quick_lookup_tables
