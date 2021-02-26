import tensorflow as tf
import wandb
from imputation import train
import yaml
import os

tfk = tf.keras
tfkl = tf.keras.layers


if __name__ == '__main__':
    # Initialise wandb
    config = yaml.load('configs/default_GTEx_inductive_imputation.yaml')  # Default config
    wandb.init(project='GTEx_imputation', config=config)
    config = wandb.config
    print(config)

    # Limit GPU
    # limit_gpu(gpu_idx=1, mem=2 * 1024)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)

    # Train model
    train(config)
