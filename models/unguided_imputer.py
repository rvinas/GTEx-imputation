import tensorflow as tf
from models.base_imputer import BaseImputer
from models.train_utils import sample_mask_tf
from data.data_utils import sample_mask
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


class UnguidedImputer(BaseImputer):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, m_low=0.5, m_high=0.5, name='UnguidedImputer', **kwargs):
        super(UnguidedImputer, self).__init__(x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name=name,
                                               **kwargs)
        self.m_low = m_low
        self.m_high = m_high

    def loss_fn(self, x, x_gen, eps=1e-7):
        x, mask = x
        # input_mask = tf.cast(x != 0, tf.float32)
        # output_mask = mask * (1 - input_mask)
        output_mask = mask

        # Masks variables that were provided as input in the forward pass
        x_ = x * output_mask  # Input variables
        x_gen_ = x_gen * output_mask  # Reconstructed input variables
        mask_counts = tf.reduce_sum(output_mask, axis=-1)  # Shape=(nb_samples, )
        loss = tf.reduce_sum(output_mask * tf.math.squared_difference(x_, x_gen_), axis=-1)

        return tf.reduce_mean(loss / (mask_counts + eps))

    def call(self, x, **kwargs):
        x, cat, num, mask = x
        if type(mask) is tuple:  # Keras is initialising
            mask = mask[0]

        x_ = x * mask
        return self.model([x_, cat, num, mask], **kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x, cat, num, mask = x

        mask, input_mask = mask
        if self.config.inplace_mode:
            output_mask = mask * (1 - input_mask)
            input_mask = mask * input_mask
        else:   # mask should be all ones
            output_mask = (1 - input_mask)

        with tf.GradientTape() as tape:
            y_pred = self.call((x, cat, num, input_mask), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss_fn((x, output_mask), y_pred)  # compiled_loss((x, mask), y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        x, cat, num, mask = x

        bs = tf.shape(x)[0]
        if self.config.inplace_mode:
            mask, input_mask = mask
            # input_mask = sample_mask(bs=mask.shape[0], nb_genes=self.x_dim)  # sample_mask_tf(bs=bs, nb_genes=self.x_dim)
            output_mask = mask * (1 - input_mask)
            input_mask = mask * input_mask
        else:
            output_mask = (1 - mask)
            input_mask = mask
        # input_mask = sample_mask(bs=self.config, nb_genes=self.x_dim)

        # Compute predictions
        y_pred = self.call((x, cat, num, input_mask), training=False)
        # Updates the metrics tracking the loss
        loss = self.loss_fn((x, output_mask), y_pred)  # compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}


