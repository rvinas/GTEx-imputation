import tensorflow as tf
import tensorflow_probability as tfp
from models.base_imputer import BaseImputer

tfk = tf.keras
tfkl = tf.keras.layers


def sample_mask_tf(bs, nb_genes, m_low=0.5, m_high=0.5):
    # Compute masks
    # m_low = 0.5?
    p_mask = tf.random.uniform(shape=(bs,), minval=m_low, maxval=m_high)  # Probability of setting mask to 0
    binomial = tfp.distributions.Binomial(1, probs=p_mask)
    mask = tf.transpose(binomial.sample(nb_genes))

    return mask


class UnguidedImputer(BaseImputer):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name='UnguidedImputer', **kwargs):
        super(UnguidedImputer, self).__init__(x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name=name,
                                               **kwargs)

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
        x_ = x * mask
        return self.model([x_, cat, num, mask], **kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x, cat, num, mask = x

        bs = tf.shape(x)[0]
        input_mask = sample_mask_tf(bs=bs, nb_genes=self.x_dim)
        output_mask = mask * (1 - input_mask)
        input_mask = mask * input_mask

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
        input_mask = sample_mask_tf(bs=bs, nb_genes=self.x_dim)
        output_mask = mask * (1 - input_mask)
        input_mask = mask * input_mask

        # Compute predictions
        y_pred = self.call((x, cat, num, input_mask), training=False)
        # Updates the metrics tracking the loss
        loss = self.loss_fn((x, output_mask), y_pred)  # compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}


