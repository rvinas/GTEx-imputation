from data.gtex_generator import GTExGenerator
from data.tcga_generator import TCGAGenerator


def get_generator(name):
    print('Dataset: {}'.format(name))
    if name == 'GTEx':
        return GTExGenerator
    elif name == 'TCGA':
        return TCGAGenerator
    else:
        raise ValueError('Unknown generator {}'.format(name))