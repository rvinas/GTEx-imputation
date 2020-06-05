import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
from collections import Counter

GTEX_FILE = 'data/GTEX_data.csv'
GTEX_FILE_CORRECTED = 'data/GTEX_CORRECTED_data.csv'
TIAGO_DATA_DIR = '/local/sdc/tmla2/gtex-analysis/data_filtered'
ENSEMBL_2_GENE_SYMBOLS = '/local/sdc/tmla2/gtex-analysis/meta_data/genes_ENSEMBL_to_official_gene.csv'
METADATA_FILE = 'data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt'


# ------------------
# GTEX
# ------------------

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


def load_gtex_unique_ENSEMBL(data_dir=TIAGO_DATA_DIR):
    # Find gene intersection
    tissues = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(tissues)

    genes = None
    for tissue in sorted(tissues):
        df = pd.read_pickle('{}/{}.pkl'.format(data_dir, tissue))
        tissue_genes = set(df.columns.values)
        if genes is None:
            genes = tissue_genes
        else:
            genes = genes.intersection(tissue_genes)
        print(len(genes))

    unique_genes = sorted(list(genes))
    print('Number of unique genes across all tissues: {}'.format(len(unique_genes)))
    return unique_genes


def load_gtex_unique_genes(prefix='only_geneids_CORRECTED_', data_dir=TIAGO_DATA_DIR):
    # Find gene intersection
    tissues = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(tissues)

    genes = None
    for tissue in sorted(tissues):
        df = pd.read_csv('{}/{}{}.csv'.format(data_dir, prefix, tissue), nrows=1, index_col=0)
        tissue_genes = set(df.columns.values)
        if genes is None:
            genes = tissue_genes
        else:
            genes = genes.intersection(tissue_genes)
        print(len(genes))

    unique_genes = sorted(list(genes))
    print('Number of unique genes across all tissues: {}'.format(len(unique_genes)))
    return unique_genes


def load_gtex_tiago(gene_symbols, prefix='only_geneids_', data_dir=TIAGO_DATA_DIR, convert_to_ENSEMBL=False):
    # gtex_files = [data_dir + f for f in os.listdir(data_dir) if f.endswith('.v8.normalized_expression.bed')]
    tissues = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(os.listdir(data_dir))

    merged_df = None
    tissues_list = []
    for tissue in sorted(tissues):
        df = pd.read_csv('{}/{}{}.csv'.format(data_dir, prefix, tissue), index_col=0)
        df = df[gene_symbols]
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.append(df)
        nb_samples_tissue = df.shape[0]
        tissues_list.extend([tissue] * nb_samples_tissue)
        # assert merged_df.columns.values[0] == 'ENSG00000000003.14'

    # Convert ENSEMBL to gene symbols
    if convert_to_ENSEMBL:
        gene_symbols, ENSEMBL_found = ENSEMBL_to_gene_symbols(gene_symbols)
        merged_df = merged_df[ENSEMBL_found]  # Discards ENSEMBL symbols not in dict
        merged_df.columns = gene_symbols
    print(merged_df.columns.values)
    print('Merged: ', merged_df.index.values)
    print(merged_df.index)

    # Append tissue column
    merged_df['tissue'] = np.array(tissues_list)
    return merged_df


def load_gtex_tiago_CORRECTED(gene_symbols, prefix='only_geneids_CORRECTED_', data_dir=TIAGO_DATA_DIR,
                              convert_to_ENSEMBL=False):
    # gtex_files = [data_dir + f for f in os.listdir(data_dir) if f.endswith('.v8.normalized_expression.bed')]
    tissues = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(os.listdir(data_dir))

    merged_df = None
    tissues_list = []
    for tissue in sorted(tissues):
        df = pd.read_csv('{}/{}{}.csv'.format(data_dir, prefix, tissue), index_col=0)
        df = df[gene_symbols]
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.append(df)
        nb_samples_tissue = df.shape[0]
        tissues_list.extend([tissue] * nb_samples_tissue)
        print('df', df.index)
        print('merged_df', merged_df.index)
        # assert merged_df.columns.values[0] == 'ENSG00000000003.14'

    # Convert ENSEMBL to gene symbols
    if convert_to_ENSEMBL:
        gene_symbols, ENSEMBL_found = ENSEMBL_to_gene_symbols(gene_symbols)
        merged_df = merged_df[ENSEMBL_found]  # Discards ENSEMBL symbols not in dict
        merged_df.columns = gene_symbols
    print(merged_df.columns.values)
    print('Merged: ', merged_df.index.values)
    print(merged_df.index)

    # Append tissue column
    merged_df['tissue'] = np.array(tissues_list)
    return merged_df


def _load_gtex(file=GTEX_FILE):
    return pd.read_csv(file, index_col=0)  # nrows=1000


def load_gtex(corrected=False):
    file = GTEX_FILE
    if corrected:
        file = GTEX_FILE_CORRECTED
    df = _load_gtex(file).sample(frac=1, random_state=0)
    tissues = df['tissue'].values
    sampl_ids = df.index.values
    del df['tissue']
    symbols = df.columns.values
    print(df.head())
    return df.values, symbols, sampl_ids, tissues


def load_gtex_mtissue_imputation(corrected=False):
    file = GTEX_FILE
    if corrected:
        file = GTEX_FILE_CORRECTED
    df = _load_gtex(file).sample(frac=1, random_state=0)
    symbols = df.columns.values
    symbols = symbols[symbols != 'tissue']
    tissues = np.array(list(sorted(set(df['tissue']))))
    nb_tissues = tissues.shape[0]
    nb_genes = symbols.shape[0]
    nb_samples = df.values.shape[0]
    tissues_count = Counter(df['tissue'])
    tissues_prop = np.array([tissues_count[t] / nb_samples for t in tissues])

    # df_ = df.groupby(df.index).apply(agg_fn)
    # print(set(df['tissue']))

    df_ = pd.pivot_table(df, values=symbols, index=[df.index, 'tissue'])
    df_ = df_.reset_index()  # .set_index('level_0')
    df_ = df_.pivot(index='level_0', columns='tissue', values=symbols)
    df_ = df_.swaplevel(0, 1, 1).sort_index(1)
    # print(df_.head(10))

    sampl_ids = df_.index.values
    tissues_ = np.unique(df_.columns.get_level_values(0))
    assert np.all(tissues == tissues_)

    nb_individuals = df_.values.shape[0]
    x = df_.values
    x = x.reshape((nb_individuals, nb_tissues, nb_genes))
    assert x[1, 1, 2] == df_.values[1, nb_genes + 2]

    return x, symbols, sampl_ids, tissues_, tissues_prop


def gtex_metadata(file=METADATA_FILE):
    df = pd.read_csv(file, delimiter='\t')
    df = df.set_index('SUBJID')
    return df


# ---------------------
# DATA UTILITIES
# ---------------------

def standardize(x):
    """
    Shape x: (nb_samples, nb_vars)
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def split_train_test(x, train_rate=0.75):
    """
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    split_point = int(train_rate * nb_samples)
    x_train = x[:split_point]
    x_test = x[split_point:]

    return x_train, x_test


def split_train_test_v2(x, sampl_ids, train_rate=0.75):
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


def save_synthetic(name, data, symbols, datadir):
    """
    Saves data with Shape=(nb_samples, nb_genes) to pickle file with the given name in datadir.
    :param name: name of the file in SYNTHETIC_DIR where the expression data will be saved
    :param data: np.array of data with Shape=(nb_samples, nb_genes)
    :param symbols: list of gene symbols matching the columns of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    data = {'data': data,
            'symbols': symbols}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_synthetic(name, datadir):
    """
    Loads data from pickle file with the given name (produced by save_synthetic function)
    :param name: name of the pickle file in datadir containing the expression data
    :return: np.array of expression with Shape=(nb_samples, nb_genes) and list of gene symbols matching the columns
    of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['symbols']


# ---------------------
# CORRELATION UTILITIES
# ---------------------

def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    return m[~np.isnan(m)]


def correlations_list(x, y, corr_fn=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fn: correlation function taking x and y as inputs
    """
    corr = corr_fn(x, y)
    return upper_diag_list(corr)


def gamma_coef(x, y):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x, x)
    dists_y = 1 - correlations_list(y, y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)
    return gamma_dx_dy


# ---------------------
# PLOTTING UTILITIES
# ---------------------

def plot_distribution(data, label, color='royalblue', linestyle='-', ax=None, plot_legend=True,
                      xlabel=None, ylabel=None):
    """
    Plot a distribution
    :param data: data for which the distribution of its flattened values will be plotted
    :param label: label for this distribution
    :param color: line color
    :param linestyle: type of line
    :param ax: matplotlib axes
    :param plot_legend: whether to plot a legend
    :param xlabel: label of the x axis (or None)
    :param ylabel: label of the y axis (or None)
    :return matplotlib axes
    """
    x = np.ravel(data)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'linestyle': linestyle, 'color': color, 'linewidth': 2, 'bw': .15},
                      label=label,
                      ax=ax)
    if plot_legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax


def plot_distance_matrix(dist_m, v_min, v_max, symbols, title='Distance matrix'):
    ax = plt.gca()
    im = ax.imshow(dist_m, vmin=v_min, vmax=v_max)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax.text(j, i, '{:.2f}'.format(dist_m[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title(title)


def plot_distance_matrices(x, y, symbols, corr_fn=pearson_correlation):
    """
    Plots distance matrices of both datasets x and y.
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :symbols: array of gene symbols. Shape=(nb_genes,)
    :param corr_fn: 2-d correlation function
    """

    dist_x = 1 - np.abs(corr_fn(x, x))
    dist_y = 1 - np.abs(corr_fn(y, y))
    v_min = min(np.min(dist_x), np.min(dist_y))
    v_max = min(np.max(dist_x), np.max(dist_y))

    # fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plot_distance_matrix(dist_x, v_min, v_max, symbols, title='Distance matrix, real')
    plt.subplot(1, 2, 2)
    plot_distance_matrix(dist_y, v_min, v_max, symbols, title='Distance matrix, synthetic')
    # fig.tight_layout()
    return plt.gca()


def plot_individual_distrs(x, y, symbols, nrows=4):
    nb_symbols = len(symbols)
    ncols = 1 + (nb_symbols - 1) // nrows

    # plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0, bottom=0, right=None, top=1.3, wspace=None, hspace=None)
    for r in range(nrows):
        for c in range(ncols):
            idx = (nrows - 1) * r + c
            plt.subplot(nrows, ncols, idx + 1)

            plt.title(symbols[idx])
            plot_distribution(x[:, idx], xlabel='', ylabel='', label='X', color='black')
            plot_distribution(y[:, idx], xlabel='', ylabel='', label='Y', color='royalblue')

            if idx + 1 == nb_symbols:
                break


def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)


def plot_tsne_2d(data, labels, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    for k, v in label_dict.items():
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    return plt.gca()


if __name__ == '__main__':
    # Merge normal tissue data
    # ENSEMBL_symbols = load_gtex_unique_ENSEMBL()
    # print(len(ENSEMBL_symbols))
    gene_symbols = load_gtex_unique_genes(prefix='only_geneids_')
    df = load_gtex_tiago(gene_symbols)
    df.to_csv(GTEX_FILE)
    # x, symbols, sampl_ids, tissues = load_gtex()
    # print(sampl_ids)
    # print(len(set(sampl_ids)))

    # Merge CORRECTED data
    # gene_symbols = load_gtex_unique_genes(prefix='only_geneids_CORRECTED_')
    # df = load_gtex_tiago_CORRECTED(gene_symbols, convert_to_ENSEMBL=False)
    # df.to_csv(GTEX_FILE_CORRECTED)
    # print(len(ENSEMBL_symbols))
    # df = load_gtex_tiago(ENSEMBL_symbols)
    # df.to_csv(GTEX_FILE)
    # x, symbols, sampl_ids, tissues = load_gtex()
    # print(sampl_ids)
    # print(len(set(sampl_ids)))

    # load_gtex_mtissue_imputation()

    # df = gtex_metadata()

    # sampl_ids = ['GTEX-1117F', 'GTEX-111CU', 'GTEX-111YS']
    # x = df.loc[sampl_ids, ['SEX', 'AGE', 'COHORT']]
