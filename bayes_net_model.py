from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from cpdmaker import make_cpd


def make_bn(training_group, edges: list):
    #  This line gives us the set of nodes.
    #  The '2' we have hardcoded in is for the two elements of each ordered pair in edges.
    nodes = {edges[i][j] for i in range(len(edges)) for j in range(2)}

    #  We need the nodes in iterable form.
    nodes = list(nodes)

    #  The ith set in parents_of_node contains the elements that have a directed edge to the ith element in nodes
    parents_of_node = [[edges[i][0] for i in range(len(edges)) if edges[i][1] == elem] for elem in nodes]

    #  the first element of 'target_and_givens' is the target-variable and the rest are the parents of target
    target_and_givens = [[node] + list(sorted(parents)) for node, parents in zip(nodes, parents_of_node)]

    #  The ith element of cardinalities is the cardinality of nodes[i].
    cardinalities = [len(training_group[node].unique()) for node in nodes]

    #  We need a dict to map nodes to their respective cardinalities for the evidence_cards.
    card_map = {node: card for node, card in zip(nodes, cardinalities)}

    evidence_cards = []
    for i in range(len(nodes)):
        card = []
        for parent in parents_of_node[i]:
            if parent is None:
                card += [None]
                break
            card += [card_map[parent]]
        evidence_cards += [card]

    '''  Let's make the BN  '''
    bn = BayesianNetwork(edges)
    for i in range(len(nodes)):
        bn.add_cpds(
            TabularCPD(nodes[i],
                       cardinalities[i],
                       values=make_cpd(training_group, *target_and_givens[i]),
                       evidence=parents_of_node[i],
                       evidence_card=evidence_cards[i]))
    return bn
