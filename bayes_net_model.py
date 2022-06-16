from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from cpdmaker import make_cpd


def make_bn(training_group, edges: list):
    #  This line gives us a list of all the nodes in the network.
    nodes = list({edges[i][j] for i in range(len(edges)) for j in range(len(edges[i]))})

    #  The ith set in parents_of_node contains the elements that have a directed edge to the ith element in nodes
    parents_of_node = [sorted([edges[i][0] for i in range(len(edges)) if edges[i][1] == node]) for node in nodes]

    #  The ith element of 'target_and_givens' is the list of target and given variables for the ith node.
    target_and_givens = [[node] + parents for node, parents in zip(nodes, parents_of_node)]

    #  The ith element of cardinalities is the cardinality of nodes[i].
    cardinalities = [len(training_group[node].unique()) for node in nodes]

    #  The ith element of parent_cards is an ordered list of the cardinalities of the parents of nodes[i]
    parent_cards = [[cardinalities[nodes.index(parent)] for parent in parents_of_node[i]] for i in range(len(nodes))]

    bn = BayesianNetwork(edges)
    for i in range(len(nodes)):
        bn.add_cpds(
            TabularCPD(nodes[i],
                       cardinalities[i],
                       values=make_cpd(training_group, *target_and_givens[i]),
                       evidence=parents_of_node[i],
                       evidence_card=parent_cards[i]))
    return bn
