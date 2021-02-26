from models.inductive_imputer import InductiveImputer
from models.unguided_imputer import UnguidedImputer
from models.gain_gtex_imputer import GAINGTEx


def get_model(name):
    if name == 'InductiveImputer':
        return InductiveImputer
    elif name == 'UnguidedImputer':
        return UnguidedImputer
    elif name == 'GAINGTEx':
        return GAINGTEx
    else:
        raise ValueError('Model {} not recognised'.format(name))
