from data.generators import get_generator
from data.eval_utils import r2_scores
import numpy as np
from tqdm import tqdm
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', dest='runs', default=3, type=int)
    parser.add_argument('--m_low', dest='m_low', default=0.5, type=float)
    parser.add_argument('--m_high', dest='m_high', default=0.5, type=float)
    parser.add_argument('--pathway', dest='pathway', default='', type=str)
    parser.add_argument('--inplace_mode', dest='inplace_mode', action='store_true')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    model = 'Median'
    pathway = args.pathway
    inplace_mode = args.inplace_mode
    nb_tissues = 49

    # Load data
    for i in range(args.runs):
        generator = get_generator('GTEx')(pathway=pathway,
                                          m_low=args.m_low,
                                          m_high=args.m_high,
                                          inplace_mode=inplace_mode,
                                          random_seed=i)
        nb_genes = generator.nb_genes
        (x, cc_, nc, mask), _ = generator.train_sample_MCAR(size=len(generator.sample_ids_train))
        if type(mask) is tuple:
            mask = mask[0]
        x = generator.x_train  # x_train without shuffling
        cc_ = generator.cat_covs_train

        x_test = np.copy(x)
        mask_zeros = np.copy(mask)
        mask[mask == 0] = np.nan
        x_nan = x * mask
        tissues = cc_[:, -1]

        mask_test_zeros = mask_zeros
        tissues_test = tissues  # Tissue label
        x_test_nan = np.copy(x_nan)
        if not inplace_mode:
            (x_test, cc_test_, nc_test, mask_test), _ = generator.test_sample_MCAR(m_low=args.m_low, m_high=args.m_high, random_seed=i)
            if type(mask_test) is tuple:
                mask_test = mask_test[0]
            x_test = generator.x_test  # x_train without shuffling
            cc_test = generator.cat_covs_test
            mask_test_zeros = np.copy(mask_test)
            mask_test[mask_test == 0] = np.nan
            x_test_nan = x_test * mask_test
            tissues_test = cc_test[:, -1]  # Tissue label

        # Impute values of patients in test set with whole blood as tissue
        t = time.time()

        x_test_imputed = []
        x_test_gt = []
        ms = []
        for tt in range(nb_tissues):
            # Find median of each gene across samples from tissue t
            idxs_tissue = np.argwhere(tissues == tt).ravel()
            medians = np.median(x[idxs_tissue], axis=0)

            # Impute missing values of test samples from tissue t
            idxs_tissue_t = np.argwhere(tissues_test == tt).ravel()
            m = mask_test_zeros[idxs_tissue_t, :]
            ms.append(m)
            x_test_gt.append((1 - m) * x_test[idxs_tissue_t, :])
            x_test_imputed.append(m * x_test[idxs_tissue_t, :] + (1 - m) * medians)

        ms = np.concatenate(ms, axis=0)
        x_test_imputed = np.concatenate(x_test_imputed, axis=0)
        x_test_gt = np.concatenate(x_test_gt, axis=0)
        t = (time.time() - t) / 3600

        # Save test loss
        x_miss = (1 - ms) * x_test_gt
        r2 = np.mean(r2_scores(x_miss, x_test_imputed, ms))

        # Save results
        name = '{}_inplace{}_{}'.format(model, inplace_mode, pathway)
        with open('results/times_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(t))
        with open('results/scores_{}.txt'.format(name), 'a') as f:
            f.write('{},'.format(r2))

        print('Model: {}, Inplace: {}, Pathway: {}, Time: {}, R2: {}'
              .format(model, inplace_mode, pathway, t, r2))
