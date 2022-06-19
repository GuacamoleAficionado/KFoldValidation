"""
Author       :    Zach Seiss
Email        :    zseiss2997@g.fmarion.edu
Written      :    June 4, 2022
Last Update  :    June 18, 2022
"""
import pandas as pd
import copy
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.inference.ExactInference import BeliefPropagation


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


def fast_query(bns: list, test_grp_indexes, environment_variables: list, data_frame: pd.DataFrame, target: str):
    inferences = [BeliefPropagation(bn) for bn in bns]
    env_map = environment_map(data_frame, environment_variables)
    quick_lookup_tables = []
    for i in range(len(inferences)):
        df = data_frame.iloc[test_grp_indexes[i]]
        groupby = df.groupby(environment_variables[:-1])[environment_variables[-1]]
        multi_index = groupby.value_counts().index
        query_evidence_table = pd.DataFrame(multi_index)

        for j in range(query_evidence_table.size):
            query_evidence = \
                {v: env_map[v][s] for v, s in zip(environment_variables,
                                                  query_evidence_table.loc[j][0])
                 if s != 'N'}
            try:
                inference = copy.deepcopy(inferences[i])
                inference = inference.query([target], query_evidence, show_progress=True)
                query_evidence_table.loc[j][0] = inference.values[1]
            except IndexError as e:
                """ For the time being if this happens we will predict 'satisfied.' """
                query_evidence_table.loc[j][0] = 1.0
                print(e)
            except ValueError as e:
                print(f'query_evidence : {query_evidence}')
                print(e)

        mymap = pd.DataFrame(range(len(multi_index)), index=multi_index)
        quick_lookup = pd.merge(mymap, query_evidence_table, left_on=mymap.columns[0], right_index=True)
        quick_lookup_tables.append(quick_lookup)

    num_queries = sum(len(e) for e in quick_lookup_tables)
    return quick_lookup_tables, num_queries, type(inferences[0])
