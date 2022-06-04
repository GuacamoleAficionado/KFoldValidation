from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from cpdmaker import make_cpd


def replace(lst, item, replacement):
    while lst.count(item) > 0:
        index = lst.index(item)
        lst.remove(item)
        lst.insert(index, replacement)


def flatten(lst):
    if not lst:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])


def make_bn(training_group, edges: list):
    #  This line gives us the set of nodes
    nodes = {edges[i][j] for i in range(len(edges)) for j in range(len(edges[i]))}

    #  We need the nodes in iterable form
    nodes = list(nodes)

    #  The ith set in directions contains the elements that have a directed edge to the ith element in nodes
    directions = [{edges[i][0] for i in range(len(edges)) if edges[i][1] == elem} for elem in nodes]

    #  'edge_map' contains key-value pairs in which the value is a tuple of nodes which have a directed
    #  edge pointing to the node given by its key.
    edge_map = {nodes[i]: sorted(list(directions[i])) for i in range(len(nodes))}

    cardinalities = [len(training_group[node].unique()) for node in nodes]

    #  we need a dict to map nodes to their respective cardinalities for the evidence_cards
    card_map = {node: card for node, card in zip(nodes, cardinalities)}
    evidence = [edge_map[nodes[i]] for i in range(len(nodes))]
    evidence = [_ if _ else None for _ in evidence]
    replace(evidence, [], [None])
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
    cpdts = []
    for i in range(len(nodes)):
        cpt_target_and_givens = []
        for k in range(len(nodes)):
            a, *b = list(edge_map.keys())[k], edge_map[list(edge_map.keys())[k]]
            b.insert(0, a)
            cpt_target_and_givens.append(flatten(b))
        if evidence[i] is not None:
            cpdts.append(
                TabularCPD(nodes[i],
                           cardinalities[i],
                           values=make_cpd(training_group, *cpt_target_and_givens[i]),
                           evidence=edge_map[nodes[i]],
                           evidence_card=evidence_cards[i])
            )
        else:
            cpdts.append(
                TabularCPD(nodes[i],
                           cardinalities[i],
                           values=make_cpd(training_group, *cpt_target_and_givens[i]))
            )
    bn.add_cpds(*cpdts)
    return bn


# print(make_bn(df, [('DenominationalGroup', 'Deactivated'),
#                    ('Deactivated', 'CongregantUsers'), ('Deactivated', 'UsingOnlineGiving'),
#                    ('Deactivated', 'Timeline')]))
