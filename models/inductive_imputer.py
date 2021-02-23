from models.base_imputer import BaseImputer
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class InductiveImputer(BaseImputer):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name='InductiveImputer', **kwargs):
        super(InductiveImputer, self).__init__(x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name=name,
                                               **kwargs)

    def loss_fn(self, x, x_gen, eps=1e-7):
        x, mask = x

        # Masks variables that were provided as input in the forward pass
        x_ = x * (1 - mask)  # Input variables
        x_gen_ = x_gen * (1 - mask)  # Reconstructed input variables
        mask_counts = tf.reduce_sum(1 - mask, axis=-1)  # Shape=(nb_samples, )
        loss = tf.reduce_sum((1 - mask) * tf.math.squared_difference(x_, x_gen_), axis=-1)

        return tf.reduce_mean(loss / (mask_counts + eps))
