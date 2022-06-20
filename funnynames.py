import random

SIZE_NOUNS = 1526
SIZE_ADJECTIVES = 1347


def get_random_file_name():
    with open('nouns.txt') as nouns, open('adjectives.txt') as adjectives:
        rn, ra = random.randint(0, SIZE_NOUNS), random.randint(0, SIZE_ADJECTIVES)
        noun = ''
        adjective = ''
        for i, line in enumerate(nouns):
            if i == rn:
                noun = line.replace('\n', '')
        for i, line in enumerate(adjectives):
            if i == ra:
                adjective = line.replace('\n', '')
    return adjective + '_' + noun
