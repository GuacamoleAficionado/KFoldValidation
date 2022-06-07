from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from cpdmaker import make_cpd


##################################  FOR TESTING  ###################################
#import pandas as pd
#df = pd.read_csv('~/PycharmProjects/KFoldValidation2/ACS_modified.csv')
####################################################################################


def make_bn(training_group, edges: list):
    #  This line gives us the set of nodes.
    nodes = {edges[i][j] for i in range(len(edges)) for j in range(len(edges[i]))}

    #  We need the nodes in iterable form.
    nodes = list(nodes)

    #  The ith set in directions contains the elements that have a directed edge to the ith element in nodes
    directions = [{edges[i][0] for i in range(len(edges)) if edges[i][1] == elem} for elem in nodes]

    #  'edge_map' contains key-value pairs in which the value is a tuple of nodes which have a directed
    #  edge pointing to the node given by its key.
    edge_map = {nodes[i]: sorted(list(directions[i])) for i in range(len(nodes))}

    #  The ith element of cardinalities is the cardinality of nodes[i].
    cardinalities = [len(training_group[node].unique()) for node in nodes]

    #  We need a dict to map nodes to their respective cardinalities for the evidence_cards.
    card_map = {node: card for node, card in zip(nodes, cardinalities)}

    #  The ith element of evidence is a list of nodes that have a directed edge 
    #  the node given by nodes[i].
    evidence = [edge_map[nodes[i]] for i in range(len(nodes))]

    #  This line just swaps out any [] in evidence for None.
    evidence = [_ if _ else None for _ in evidence]

    #  The ith element of evidence_cards is a list of cardinalities that
    #  correspond to the ith list in evidence.
    evidence_cards = []
    for i in range(len(evidence)):
        if evidence[i] is None:
            evidence_cards += [None]
            continue
        card = []
        for j in range(len(evidence[i])):
            card += [card_map[evidence[i][j]]]
        evidence_cards += [card]


    '''  Let's make the BN  '''

    bn = BayesianNetwork(edges)
    for i in range(len(nodes)):
        key = list(edge_map.keys())[i]
        target_and_givens = [key] + edge_map[key]
        bn.add_cpds(
            TabularCPD(nodes[i],
                       cardinalities[i],
                       values=make_cpd(training_group, *target_and_givens),
                       evidence=evidence[i],
                       evidence_card=evidence_cards[i]))
    return bn
