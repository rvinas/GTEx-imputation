import numpy as np
import pandas as pd
from data.data_utils import standardize, split_train_test, sample_mask, ENSEMBL_to_gene_symbols
from data.pathways import select_genes_pathway


def TCGA_FILE(cancer_type):
    return '/local/scratch/rv340/tcga/TCGA-{}.htseq_fpkm.tsv'.format(cancer_type)


def TCGA_METADATA_FILE(cancer_type):
    return '/local/scratch/rv340/tcga/{}_clinicalMatrix'.format(cancer_type)


def get_GTEx_tissue(cancer_type):
    if cancer_type == 'LAML':
        return 'Whole_Blood', 48
    elif cancer_type == 'BRCA':
        return 'Breast_Mammary_Tissue', 19
    elif cancer_type == 'LUAD':
        return 'Lung', 31
    else:
        raise ValueError('Cancer type {} not supported'.format(cancer_type))


def TCGA(file, clinical_file, tissue_idx=None, gtex_gene_symbols=None):
    df = pd.read_csv(file, delimiter='\t')
    df = df.set_index('Ensembl_ID')

    # Transform gene symbols
    gene_symbols, ENSEMBL_found = ENSEMBL_to_gene_symbols(df.index)
    df = df.loc[ENSEMBL_found]
    df = df.rename(index=dict(zip(df.index, gene_symbols)))
    if gtex_gene_symbols is not None:
        df = df.loc[gtex_gene_symbols]
        gene_symbols = gtex_gene_symbols
    df = df.groupby('Ensembl_ID', group_keys=False).apply(
        lambda x: x[x.sum(axis=1) == np.max(x.sum(axis=1))])  # Remove duplicates, keep max

    # Get data
    x_TCGA = df.values.T

    # Process covariates
    sample_ids = df.columns
    clinical_df = pd.read_csv(clinical_file, delimiter='\t')
    idxs = [np.argwhere(s[:-1] == clinical_df['sampleID']).ravel()[0] for s in df.columns]
    gender = np.array([0 if g == 'MALE' else 1 for g in clinical_df.iloc[idxs]['gender']])
    age = clinical_df.iloc[idxs]['age_at_initial_pathologic_diagnosis'].values
    mean_age = 52.7763  # Mean age GTEx
    std_age = 12.9351  # Std age GTEx
    age = (age - mean_age) / std_age
    cc_TCGA = np.zeros((x_TCGA.shape[0], 3))
    cc_TCGA[:, 0] = gender
    cc_TCGA[:, 2] = tissue_idx
    nc_TCGA = age[..., None]

    return x_TCGA, gene_symbols, sample_ids, np.int32(cc_TCGA), nc_TCGA


class TCGAGenerator:
    def __init__(self, pathway=None, cancer_type=None, file=None, metadata_file=None,
                 gene_symbols=None, batch_size=128, m_low=0.5, m_high=0.5, random_seed=0):
        assert cancer_type or (file and metadata_file)
        if file is None:
            file = TCGA_FILE(cancer_type)
        if metadata_file is None:
            metadata_file = TCGA_METADATA_FILE(cancer_type)
        self.tissue = None
        self.tissue_idx = None
        if cancer_type:
            self.tissue, self.tissue_idx = get_GTEx_tissue(cancer_type)

        np.random.seed(random_seed)
        self.file = file
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.m_low = m_low
        self.m_high = m_high
        self.pathway = pathway

        # Load data
        x_TCGA, gene_symbols, sample_ids, cc_TCGA, nc_TCGA = TCGA(file, metadata_file, tissue_idx=self.tissue_idx,
                                                                  gtex_gene_symbols=gene_symbols)

        # Select genes from specific pathway
        gene_idxs, self.gene_symbols = select_genes_pathway(gene_symbols, pathway)
        self.x = standardize(np.log(1 + x_TCGA[:, gene_idxs]))
        self.x[np.isnan(self.x)] = 0  # Genes with std 0
        self.nb_genes = len(self.gene_symbols)

        # Store covariates
        self.cat_covs = cc_TCGA
        self.num_covs = nc_TCGA

    def train_sample(self, size=None):
        if size is None:
            size = self.batch_size
        sample_idxs = np.random.choice(self.x.shape[0], size=size, replace=False)
        x = self.x[sample_idxs]
        cc = self.cat_covs[sample_idxs]
        nc = self.num_covs[sample_idxs]
        return x, cc, nc

    def train_sample_MCAR(self, size=None, m_low=None, m_high=None):
        if size is None:
            size = self.batch_size
        if m_low is None:
            m_low = self.m_low
        if m_high is None:
            m_high = self.m_high
        mask = sample_mask(size, self.nb_genes, m_low=m_low, m_high=m_high)
        x, cc, nc = self.train_sample(size)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x

    def train_iterator_MCAR(self):
        while True:
            yield self.train_sample_MCAR()

    def val_sample(self):
        x = self.x
        cc = self.cat_covs
        nc = self.num_covs
        return x, cc, nc

    def val_sample_MCAR(self, m_low=None, m_high=None):
        if m_low is None:
            m_low = self.m_low
        if m_high is None:
            m_high = self.m_high
        x, cc, nc = self.val_sample()
        size = x.shape[0]
        mask = sample_mask(size, self.nb_genes, m_low=m_low, m_high=m_high)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x

    def test_sample(self):
        x = self.x
        cc = self.cat_covs
        nc = self.num_covs
        return x, cc, nc

    def test_sample_MCAR(self, m_low=None, m_high=None, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        if m_low is None:
            m_low = self.m_low
        if m_high is None:
            m_high = self.m_high
        x, cc, nc = self.test_sample()
        size = x.shape[0]
        mask = sample_mask(size, self.nb_genes, m_low=m_low, m_high=m_high)
        # x_ = mask * x
        # y = (1 - mask) * x
        return (x, cc, nc, mask), x
