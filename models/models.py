from models.inductive_imputer import InductiveImputer
from models.unguided_imputer import UnguidedImputer
from models.gain_gtex_imputer import GAINGTEx
from models.gain_mse_gtex_imputer import GAINMSEGTEx


def get_model(name):
    if name == 'InductiveImputer':
        return InductiveImputer
    elif name == 'UnguidedImputer':
        return UnguidedImputer
    elif name == 'GAINGTEx':
        return GAINGTEx
    elif name == 'GAINMSEGTEx':
        return GAINMSEGTEx
    else:
        raise ValueError('Model {} not recognised'.format(name))
