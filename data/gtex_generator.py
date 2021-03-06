import numpy as np
import pandas as pd
from data.data_utils import standardize, split_train_test, sample_mask
from data.pathways import select_genes_pathway

GTEX_FILE = '/home/rv340/adversarial-gtex/data/GTEX_data.csv'
METADATA_FILE = '/home/rv340/adversarial-gtex/data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt'


def GTEx(file, random_seed=0):
    df = pd.read_csv(file, index_col=0).sample(frac=1, random_state=random_seed)
    tissues = df['tissue'].values
    sampl_ids = df.index.values
    del df['tissue']
    gene_symbols = df.columns.values
    return np.float32(df.values), gene_symbols, sampl_ids, tissues


def GTEx_metadata(file):
    df = pd.read_csv(file, delimiter='\t')
    df = df.set_index('SUBJID')
    return df


class GTExGenerator:
    def __init__(self, file=GTEX_FILE, metadata_file=METADATA_FILE, pathway=None, batch_size=128, m_low=0.5, m_high=0.5,
                 random_seed=0, inplace_mode=False):
        np.random.seed(random_seed)
        self.file = file
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.m_low = m_low
        self.m_high = m_high
        self.pathway = pathway
        self.inplace_mode = inplace_mode

        # Load data
        x, gene_symbols, self.sample_ids, self.tissues = GTEx(file)

        # Select genes from specific pathway
        gene_idxs, self.gene_symbols = select_genes_pathway(gene_symbols, pathway)
        self.x = x[:, gene_idxs]
        self.nb_genes = len(self.gene_symbols)

        # Load metadata
        df_metadata = GTEx_metadata(metadata_file)
        self.metadata = df_metadata

        # Process categorical metadata
        cat_cols = ['SEX', 'COHORT']  # 'SEX', 'COHORT'
        self.cat_cols = cat_cols
        df_metadata[cat_cols] = df_metadata[cat_cols].astype('category')
        cat_dicts = [df_metadata[cat_col].cat.categories.values for cat_col in cat_cols]
        df_metadata[cat_cols] = df_metadata[cat_cols].apply(lambda x: x.cat.codes)
        cat_covs = df_metadata.loc[self.sample_ids, cat_cols].values
        tissues_dict_inv = np.array(list(sorted(set(self.tissues))))
        tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
        tissues = np.vectorize(lambda t: tissues_dict[t])(self.tissues)
        cat_dicts.append(tissues_dict_inv)
        cat_covs = np.concatenate((cat_covs, tissues[:, None]), axis=-1)
        cat_covs = np.int32(cat_covs)
        self.tissues_dict = tissues_dict
        self.tissues_dict_inv = tissues_dict_inv
        self.vocab_sizes = [len(c) for c in cat_dicts]
        self.nb_categorical = cat_covs.shape[-1]

        # Process numerical metadata
        num_cols = ['AGE']  # 'AGE'
        num_covs = df_metadata.loc[self.sample_ids, num_cols].values
        num_covs = standardize(num_covs)
        num_covs = np.float32(num_covs)
        self.nb_numeric = num_covs.shape[-1]

        # Train/val/test split
        x_train, x_test, sampl_ids_train, sampl_ids_test = split_train_test(self.x, self.sample_ids)
        x_train = standardize(x_train)
        x_test = standardize(x_test)
        x_train, x_val, _, sampl_ids_val = split_train_test(x_train, sampl_ids_train, train_rate=0.8)
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test

        num_covs_train, num_covs_test, _, _ = split_train_test(num_covs, self.sample_ids)
        num_covs_train = standardize(num_covs_train)
        num_covs_test = standardize(num_covs_test)
        num_covs_train, num_covs_val, _, _ = split_train_test(num_covs_train, sampl_ids_train, train_rate=0.8)
        self.num_covs_train = num_covs_train
        self.num_covs_val = num_covs_val
        self.num_covs_test = num_covs_test

        cat_covs_train, cat_covs_test, _, _ = split_train_test(cat_covs, self.sample_ids)
        cat_covs_train, cat_covs_val, sampl_ids_train, sampl_ids_val = split_train_test(cat_covs_train,
                                                                                        sampl_ids_train,
                                                                                        train_rate=0.8)
        self.cat_covs_train = cat_covs_train
        self.cat_covs_val = cat_covs_val
        self.cat_covs_test = cat_covs_test

        self.sample_ids_train = sampl_ids_train
        self.sample_ids_val = sampl_ids_val
        self.sample_ids_test = sampl_ids_test

        self.train_mask = sample_mask(len(sampl_ids_train), self.nb_genes, m_low=m_low, m_high=m_high)
        self.val_mask = sample_mask(len(sampl_ids_val), self.nb_genes, m_low=m_low, m_high=m_high)
        self.test_mask = sample_mask(len(sampl_ids_test), self.nb_genes, m_low=m_low, m_high=m_high)




    """
    def train_sample(self, size=None):
        if size is None:
            size = self.batch_size
        sample_idxs = np.random.choice(self.x_train.shape[0], size=size, replace=False)
        x = self.x_train[sample_idxs]
        cc = self.cat_covs_train[sample_idxs]
        nc = self.num_covs_train[sample_idxs]
        return x, cc, nc
    """

    def train_sample_MCAR(self, size=None, alpha=0.5, beta=0.5):
        if size is None:
            size = self.batch_size
        sample_idxs = np.random.choice(self.x_train.shape[0], size=size, replace=False)
        x = self.x_train[sample_idxs]
        cc = self.cat_covs_train[sample_idxs]
        nc = self.num_covs_train[sample_idxs]
        mask_2 = sample_mask(size, self.nb_genes, m_low=alpha, m_high=beta)
        if self.inplace_mode:
            mask_1 = self.train_mask[sample_idxs]
        else:
            mask_1 = sample_mask(size, self.nb_genes, m_low=self.m_low, m_high=self.m_high)
        mask = (mask_1, mask_2)
        # x, cc, nc = self.train_sample(size)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x

    def train_iterator_MCAR(self, alpha=0.5, beta=0.5):
        while True:
            yield self.train_sample_MCAR(size=self.batch_size, alpha=alpha, beta=beta)

    def val_sample(self):
        x = self.x_val
        cc = self.cat_covs_val
        nc = self.num_covs_val
        return x, cc, nc

    def val_sample_MCAR(self, alpha=0.5, beta=0.5):
        x, cc, nc = self.val_sample()
        # size = x.shape[0]
        if self.inplace_mode:
            input_mask = sample_mask(x.shape[0], self.nb_genes, m_low=alpha, m_high=beta)
            mask = (self.val_mask, input_mask)  # Trick to speed up training
        else:
            mask = sample_mask(x.shape[0], self.nb_genes, m_low=self.m_low, m_high=self.m_high)
        # mask = sample_mask(size, self.nb_genes, m_low=m_low, m_high=m_high)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x

    def test_sample(self):
        x = self.x_test
        cc = self.cat_covs_test
        nc = self.num_covs_test
        return x, cc, nc

    def test_sample_MCAR(self, m_low=0.5, m_high=0.5, random_seed=0):
        if self.inplace_mode:
            return self.train_sample_MCAR(size=len(self.sample_ids_train))

        np.random.seed(random_seed)
        x, cc, nc = self.test_sample()
        size = x.shape[0]
        mask = sample_mask(size, self.nb_genes, m_low=m_low, m_high=m_high)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x