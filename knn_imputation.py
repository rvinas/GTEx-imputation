from data.generators import get_generator
from data.eval_utils import r2_scores
from data.data_utils import get_dummies
from sklearn.impute import KNNImputer
import numpy as np
from missingpy import MissForest
import argparse
import time
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', dest='k', default=1, type=int)
    parser.add_argument('--runs', dest='runs', default=3, type=int)
    parser.add_argument('--m_low', dest='m_low', default=0.5, type=float)
    parser.add_argument('--m_high', dest='m_high', default=0.5, type=float)
    parser.add_argument('--pathway', dest='pathway', default='', type=str)
    parser.add_argument('--inplace_mode', dest='inplace_mode', action='store_true')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    model = 'KNN_k{}'.format(args.k)
    pathway = args.pathway
    inplace_mode = args.inplace_mode

    # Load data
    for i in range(args.runs):
        generator = get_generator('GTEx')(pathway=pathway,
                                          m_low=args.m_low,
                                          m_high=args.m_high,
                                          inplace_mode=inplace_mode,
                                          random_seed=i)
        (x, cc_, nc, mask), _ = generator.train_sample_MCAR(size=len(generator.sample_ids_train))
        x_test = np.copy(x)
        cc = get_dummies(cc_)
        mask_zeros = np.copy(mask)
        mask[mask == 0] = np.nan
        x_nan = x * mask
        x_input = np.concatenate((x_nan, cc, nc), axis=-1)

        x_test_input = x_input
        mask_test_zeros = mask_zeros
        if not inplace_mode:
            (x_test, cc_test_, nc_test, mask_test), _ = generator.test_sample_MCAR(m_low=args.m_low, m_high=args.m_high)
            cc_test = get_dummies(np.concatenate((cc_, cc_test_), axis=0))[-x_test.shape[0]:]
            mask_test_zeros = np.copy(mask_test)
            mask_test[mask_test == 0] = np.nan
            x_test_nan = x_test * mask_test
            x_test_input = np.concatenate((x_test_nan, cc_test, nc_test), axis=-1)

        # Train model
        t = time.time()
        imputer = KNNImputer(n_neighbors=args.k)
        imputer.fit(x_input)
        x_imp = imputer.transform(x_test_input)

        x_imp = x_imp[:, :generator.nb_genes]
        t = (time.time() - t) / 3600

        # Save test loss
        x_miss = (1 - mask_test_zeros) * x_test
        r2 = np.mean(r2_scores(x_miss, x_imp, mask_test_zeros))

        # Save results
        name = '{}_inplace{}_{}'.format(model, inplace_mode, pathway)
        with open('results/times_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(t))
        with open('results/scores_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(r2))

        print('Model: {}, Inplace: {}, Pathway: {}, Time: {}, R2: {}'
              .format(model, inplace_mode, pathway, t, r2))



