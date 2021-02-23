import numpy as np
from data.data_utils import sample_mask
import matplotlib.pyplot as plt
import seaborn as sns


def extend_dataset(x_test, cat_covs_test, num_covs_test, tissues_test,
                   n=171 * 2, nb_tissues=49, m_low=0.5, m_high=0.5):
    # Create extended test dataset (so that all tissues have same number of test samples)
    nb_test, nb_genes = x_test.shape
    x_test_extended = np.empty((nb_tissues * n, nb_genes), dtype=np.float32)
    cat_covs_test_extended = np.empty((nb_tissues * n, cat_covs_test.shape[-1]), dtype=np.int32)
    num_covs_test_extended = np.empty((nb_tissues * n, num_covs_test.shape[-1]), dtype=np.float32)

    for t in range(nb_tissues):
        idxs_tissue = np.argwhere(tissues_test == t).ravel()
        idxs_extended = np.resize(idxs_tissue, n)
        x_test_extended[t * n: (t + 1) * n] = x_test[idxs_extended]
        cat_covs_test_extended[t * n: (t + 1) * n] = cat_covs_test[idxs_extended]
        num_covs_test_extended[t * n: (t + 1) * n] = num_covs_test[idxs_extended]

    # Create mask
    mask = sample_mask(x_test_extended.shape[0], nb_genes, m_low=m_low, m_high=m_high)

    return x_test_extended, cat_covs_test_extended, num_covs_test_extended, mask


def r2_scores(x_gt, x_pred, mask):
    mask_r = np.copy(mask)
    mask_r[mask_r == 1] = np.nan

    gene_means = np.mean(x_gt, axis=0)  # Shape=(nb_genes,)
    mask_r[:, gene_means == 0] = np.nan  # Discard genes with 0 variance
    ss_res = np.nansum((1 - mask_r) * (x_gt - x_pred) ** 2, axis=0)
    ss_tot = np.nansum((1 - mask_r) * (x_gt - gene_means) ** 2, axis=0)
    r_sq = 1 - ss_res / ss_tot
    return r_sq


def r2_scores_3d(x_test_extended, x_gen_extended, mask, nb_tissues=49):
    # Assumes inputs with shape (nb_tissues * nb_samples, nb_genes)
    # Compute R^2
    nb_genes = x_test_extended.shape[-1]
    x_test_extended_r = x_test_extended.reshape((nb_tissues, -1, nb_genes))
    assert x_test_extended_r[0, 2, 0] == x_test_extended[2, 0]

    x_gen_extended_r = x_gen_extended.reshape(
        (nb_tissues, -1, nb_genes))  # Shape=(nb_tissues, nb_samples_tissue, nb_genes)
    mask_r = np.copy(mask.reshape((nb_tissues, -1, nb_genes)))
    mask_r[mask_r == 1] = np.nan

    gene_means = np.mean(x_test_extended_r, axis=1)  # Shape=(nb_tissues, nb_genes)

    ss_res = np.nansum((1 - mask_r) * (x_test_extended_r - x_gen_extended_r) ** 2, axis=1)
    ss_tot = np.nansum((1 - mask_r) * (x_test_extended_r - gene_means[:, None, :]) ** 2, axis=1)

    r_sq = 1 - ss_res / ss_tot
    return r_sq


def mse_scores(x_gt, x_pred, mask):
    return np.sum((1 - mask) * (x_pred - x_gt) ** 2) / np.sum(1 - mask)


def mse_scores_3d(x_test_extended, x_gen_extended, mask, nb_tissues=49):
    # Assumes inputs with shape (nb_tissues * nb_samples, nb_genes)
    nb_genes = x_test_extended.shape[-1]
    x_test_extended_r = x_test_extended.reshape((nb_tissues, -1, nb_genes))
    assert x_test_extended_r[0, 2, 0] == x_test_extended[2, 0]

    x_gen_extended_r = x_gen_extended.reshape(
        (nb_tissues, -1, nb_genes))  # Shape=(nb_tissues, nb_samples_tissue, nb_genes)
    mask_r = np.copy(mask.reshape((nb_tissues, -1, nb_genes)))
    mask_r[mask_r == 1] = np.nan
    err = (1 - mask_r) * (x_gen_extended_r - x_test_extended_r) ** 2

    return np.nanmean(err, axis=1)


#################################
# PLOTTING UTILS
#################################

def box_plot_scores(r_sq, nb_train, tissues_dict_inv):
    colors = plt.cm.rainbow(nb_train.flatten())
    sns.boxplot(x=np.array([[t] * r_sq.shape[1] for t in tissues_dict_inv]).flatten(), y=r_sq.flatten(),
                sym='')
    ax = plt.gca()

    cmap = plt.cm.summer  # Greens #Blues # viridis
    colors = cmap(nb_train.flatten())
    for i, a in enumerate(ax.artists):
        # Change the appearance of that box
        a.set_facecolor(colors[i])
        # a.set_edgecolor('black')
        # a.set_linewidth(3)

    norm = plt.Normalize(nb_train.min(), nb_train.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.xticks(rotation=90)

    # ax.figure.colorbar(colors)
    ax.set_ylim([0, 1])
    # plt.title('$R^2$ imputation scores per tissue')
    return plt.gca()

