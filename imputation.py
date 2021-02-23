from models.models import get_model
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from data.generators import get_generator
from models.inductive_imputer import InductiveImputer
import yaml
import os

tfk = tf.keras
tfkl = tf.keras.layers


def train(config):
    # Load data
    generator = get_generator(config.dataset)(pathway=config.pathway,
                                              batch_size=config.batch_size,
                                              m_low=config.m_low,
                                              m_high=config.m_high)

    # Make model
    model = get_model(config.model)(x_dim=generator.nb_genes,
                                    vocab_sizes=generator.vocab_sizes,
                                    nb_numeric=generator.nb_numeric,
                                    nb_categoric=generator.nb_categorical,
                                    config=config)
    opt = tf.keras.optimizers.Adam(config.lr)
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

    model.fit(generator.train_iterator_MCAR(),
              validation_data=generator.val_sample_MCAR(),
              epochs=config.epochs,
              steps_per_epoch=config.steps_per_epoch,
              callbacks=[WandbCallback(), early_stopper])
    model.save('{}/checkpoints/{}_{}'.format(config.save_dir, config.model, config.pathway))

    return model, generator


if __name__ == '__main__':
    # Initialise wandb
    config = yaml.load('configs/default_GTEx_inductive_imputation.yaml')  # Default config
    run = wandb.init(project='GTEx_imputation', config=config)
    config = wandb.config
    print(config)

    # Limit GPU
    # limit_gpu(gpu_idx=1, mem=2 * 1024)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)

    # Train model
    model, generator = train(config)
