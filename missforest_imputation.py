from data.generators import get_generator
from data.eval_utils import r2_scores
from data.data_utils import get_dummies
import numpy as np
from missingpy import MissForest
import argparse
import time
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', dest='nt', default=1, type=int)
    parser.add_argument('--runs', dest='runs', default=3, type=int)
    parser.add_argument('--m_low', dest='m_low', default=0.5, type=float)
    parser.add_argument('--m_high', dest='m_high', default=0.5, type=float)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    model = 'MissForest_{}trees'.format(args.nt)
    pathway = 'Alzheimer'
    inplace_mode = True

    # Load data
    for i in range(args.runs):
        generator = get_generator('GTEx')(pathway=pathway,
                                          m_low=args.m_low,
                                          m_high=args.m_high,
                                          inplace_mode=inplace_mode,
                                          random_seed=i)
        (x, cc, nc, mask), _ = generator.train_sample_MCAR(size=len(generator.sample_ids_train))
        cc = get_dummies(cc)
        mask_zeros = np.copy(mask)
        mask[mask == 0] = np.nan
        x_nan = x * mask
        x_input = np.concatenate((x_nan, cc, nc), axis=-1)

        # Train model
        t = time.time()
        imputer = MissForest(bootstrap=True, class_weight=None, copy=True,
                             criterion=('mse', 'gini'), decreasing=False, max_depth=None,
                             max_features='auto', max_iter=10, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, missing_values=np.nan, n_estimators=args.nt,
                             n_jobs=10, oob_score=False, random_state=None, verbose=0,
                             warm_start=False)
        x_imp = imputer.fit_transform(x_input)
        x_imp = x_imp[:, :generator.nb_genes]
        t = (time.time() - t) / 3600

        # Save test loss
        x_miss = (1 - mask_zeros) * x
        r2 = np.mean(r2_scores(x_miss, x_imp, mask_zeros))

        # Save results
        name = '{}_inplace{}_{}'.format(model, inplace_mode, pathway)
        with open('results/times_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(t))
        with open('results/scores_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(r2))

        print('Model: {}, Inplace: {}, Pathway: {}, Time: {}, R2: {}'
              .format(model, inplace_mode, pathway, t, r2))



