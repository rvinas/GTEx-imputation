import numpy as np
import pandas as pd

ENSEMBL_2_GENE_SYMBOLS = '/local/scratch/rv340/hugo/genes_ENSEMBL_to_official_gene.csv'


def ENSEMBL_to_gene_symbols(ENSEMBL_symbols, file=ENSEMBL_2_GENE_SYMBOLS):
    def _ENSEMBL_to_gene_symbols(file):
        df = pd.read_csv(file, header=None)
        df.columns = ['ensemble', 'doubt', 'geneid']
        # e2gs = {e: gid for e, gid in zip(df['ensemble'], df['geneid'])}
        df.set_index('ensemble', inplace=True)
        df.drop(columns=['doubt'], inplace=True)
        return df

    # ENSEMBL to gene names
    e2gs = _ENSEMBL_to_gene_symbols(file)
    gene_symbols = []
    ENSEMBL_found = []
    for g in ENSEMBL_symbols:
        gs = e2gs.geneid.get(g.split('.')[0], None)
        if gs is not None:
            ENSEMBL_found.append(g)
            gene_symbols.append(gs)

    return np.array(gene_symbols), np.array(ENSEMBL_found)


def standardize(x):
    """
    Shape x: (nb_samples, nb_vars)
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def split_train_test(x, sampl_ids, train_rate=0.75):
    """
    Avoids patient leak between train/test set
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    sample_ids_rev = np.array([s[::-1] for s in sampl_ids])
    split_point = int(train_rate * nb_samples)
    idxs = np.argsort(sample_ids_rev)
    sample_ids_rev_sorted = sample_ids_rev[idxs]

    p = split_point
    while p == nb_samples and sample_ids_rev_sorted[p - 1] == sample_ids_rev_sorted[p - 2]:
        p += 1
    if p == nb_samples:
        raise Exception('Error: Cannot split samples into train and test sets')

    x_train = x[idxs[:p]]
    x_test = x[idxs[p:]]
    sample_ids_train = sampl_ids[idxs[:p]]
    sample_ids_test = sampl_ids[idxs[p:]]

    return x_train, x_test, sample_ids_train, sample_ids_test


def sample_mask(bs, nb_genes, m_low=0.5, m_high=0.95):
    # Compute masks
    # m_low = 0.5?
    p_mask = np.random.uniform(low=m_low, high=m_high, size=(bs,))  # Probability of setting mask to 0
    mask = np.random.binomial(1, p_mask, size=(nb_genes, bs)).astype(np.float32).T  # Shape=(bs, nb_genes)

    return mask
