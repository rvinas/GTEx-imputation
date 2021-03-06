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
    model = 'BloodSurrogate'
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
        sample_ids_test = generator.sample_ids_train
        if not inplace_mode:
            (x_test, cc_test_, nc_test, mask_test), _ = generator.test_sample_MCAR(m_low=args.m_low, m_high=args.m_high,
                                                                                   random_seed=i)
            if type(mask_test) is tuple:
                mask_test = mask_test[0]
            x_test = generator.x_test  # x_train without shuffling
            cc_test = generator.cat_covs_test
            mask_test_zeros = np.copy(mask_test)
            mask_test[mask_test == 0] = np.nan
            x_test_nan = x_test * mask_test
            tissues_test = cc_test[:, -1]  # Tissue label
            sample_ids_test = generator.sample_ids_test

        # Impute values of patients in test set with whole blood as tissue
        t = time.time()

        blood_idx = generator.tissues_dict['Whole_Blood']
        blood_idxs = np.argwhere(tissues_test == blood_idx).ravel()
        patients_with_blood_tissue = sample_ids_test[blood_idxs]
        print('# patients: ', len(patients_with_blood_tissue))

        x_test_imputed = []
        x_test_gt = []
        ms = []
        for idx in tqdm(blood_idxs):
            patient_id = sample_ids_test[idx]
            blood_expr = x[idx, :]
            samples_patient = [j for j, s in enumerate(sample_ids_test) if s == patient_id]

            for s in samples_patient:
                m = mask_test_zeros[s]
                ms.append(m)
                x_test_imputed.append((1 - m) * blood_expr)
                x_test_gt.append((1 - m) * x_test[s, :])
                # print('Real: ', m * x_test[s, :])
                # print('Imp: ', m * x_test[s, :] + (1 - m) * blood_expr)

        ms = np.array(ms)
        x_test_imputed = np.array(x_test_imputed)
        x_test_gt = np.array(x_test_gt)
        print('# imputed samples: ', x_test_imputed.shape[0])

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
