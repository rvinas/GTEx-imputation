from models.inductive_imputer import InductiveImputer
from models.unguided_imputer import UnguidedImputer


def get_model(name):
    if name == 'InductiveImputer':
        return InductiveImputer
    elif name == 'UnguidedImputer':
        return UnguidedImputer
    else:
        raise ValueError('Model {} not recognised'.format(name))
