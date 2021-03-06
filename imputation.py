from models.models import get_model
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from data.generators import get_generator
from data.eval_utils import r2_scores
import numpy as np
import argparse
import yaml
import time
import os

tfk = tf.keras
tfkl = tf.keras.layers


def train(config):
    # Load data
    generator = get_generator(config.dataset)(pathway=config.pathway,
                                              batch_size=config.batch_size,
                                              m_low=config.m_low,
                                              m_high=config.m_high,
                                              inplace_mode=config.inplace_mode,
                                              random_seed=config.random_seed)

    # Make model
    model = get_model(config.model)(x_dim=generator.nb_genes,
                                    vocab_sizes=generator.vocab_sizes,
                                    nb_numeric=generator.nb_numeric,
                                    nb_categoric=generator.nb_categorical,
                                    config=config)
    opt = tf.keras.optimizers.Adam(config.lr)
    if config.model == 'GAINGTEx':
        disc_opt = tf.keras.optimizers.Adam(config.lr)
        opt = (opt, disc_opt)
    model.compile(opt)

    # Test save
    # model.save('checkpoints/inductive_imputer')

    # Train model
    early_stopper = tfk.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=config.patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    alpha = 0.5
    beta = 0.5
    if 'alpha' in config:
        alpha = config.alpha
    elif 'beta' in config:
        beta = config.beta
    model.fit(generator.train_iterator_MCAR(alpha=alpha, beta=beta),
              validation_data=generator.val_sample_MCAR(alpha=alpha, beta=beta),
              epochs=config.epochs,
              steps_per_epoch=config.steps_per_epoch,
              callbacks=[WandbCallback(), early_stopper])
    model.save(
        '{}/checkpoints/{}_inplace{}_{}'.format(config.save_dir, config.model, config.inplace_mode, config.pathway))

    return model, generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default_GTEx_inductive_imputation.yaml', type=str)
    parser.add_argument('--random_seed', dest='random_seed', default=0, type=int)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    wandb.init(project='GTEx_imputation', config=args.config)
    wandb.config.update({'random_seed': args.random_seed}, allow_val_change=True)
    config = wandb.config
    print(config)

    # Limit GPU
    # limit_gpu(gpu_idx=1, mem=2 * 1024)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)

    # Train model
    t = time.time()
    model, generator = train(config)
    t = (time.time() - t) / 3600

    # Save test loss
    x, _ = generator.test_sample_MCAR()
    x, cc, nc, mask = x
    if type(mask) is tuple:
        mask = mask[0]
    x_obs = mask * x
    x_miss = (1 - mask) * x
    x_imp = model((x_obs, cc, nc, mask))  # imputer.impute((x_observed, cc, nc, mask))
    r2 = np.mean(r2_scores(x_miss, x_imp, mask))

    # Save results
    name = '{}_inplace{}_{}'.format(config.model, config.inplace_mode, config.pathway)
    with open('results/times_{}.txt'.format(name), 'a') as f:
        f.write('{},'.format(t))
    with open('results/scores_{}.txt'.format(name), 'a') as f:
        f.write('{},'.format(r2))

    print('Model: {}, Inplace: {}, Pathway: {}, Time: {}, R2: {}'
          .format(config.model, config.inplace_mode, config.pathway, t, r2))
