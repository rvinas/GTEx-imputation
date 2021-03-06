import tensorflow as tf
import tensorflow_probability as tfp
from models.base_imputer import BaseImputer
from models.train_utils import sample_mask_tf
from models.gain_gtex_imputer import make_generator
from data.data_utils import sample_mask

tfk = tf.keras
tfkl = tf.keras.layers




class GAINMSEGTEx(BaseImputer):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, m_low=0.5, m_high=0.5, name='GAINMSEGTEx', **kwargs):
        super(GAINMSEGTEx, self).__init__(x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name=name,
                                               **kwargs)
        self.m_low = m_low
        self.m_high = m_high
        self.model = self.gen

    def _create_model(self):
        self.gen = make_generator(x_dim=self.x_dim,
                                  vocab_sizes=self.vocab_sizes,
                                  z_dim=self.x_dim,
                                  nb_numeric=self.nb_numeric,
                                  nb_layers=self.config['nb_layers'],
                                  hdim=self.config['hdim'],
                                  bn=self.config['bn'],
                                  dropout=self.config['dropout'])
        return None


    def call(self, x, **kwargs):
        x, cat, num, mask = x
        if type(mask) is tuple:  # Keras is initialising
            mask = mask[0]

        x_ = x * mask
        bs = tf.shape(x)[0]
        z = tf.random.normal([bs, self.x_dim])
        z = z * (1 - mask)
        return self.gen([x_, z, cat, num, mask], **kwargs)

    def compile(self, optimizer):
        super(BaseImputer, self).compile()
        self.gen_opt = optimizer

    def supervised_loss(self, x, x_gen, mask, eps=1e-7):
        # Masks variables that were discarded as input in the forward pass
        x_ = x * mask  # Input variables
        x_gen_ = x_gen * mask  # Reconstructed input variables
        mask_counts = tf.reduce_sum(mask, axis=-1)  # Shape=(nb_samples, )
        loss = tf.reduce_sum(mask * tf.math.squared_difference(x_, x_gen_), axis=-1)

        return tf.reduce_mean(loss / (mask_counts + eps))

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x, cat, num, mask = x

        # bs = tf.shape(x)[0]
        # if not self.config.inplace_mode:
        # mask = sample_mask_tf(bs=bs, nb_genes=self.x_dim, m_low=self.m_low, m_high=self.m_high)
        # mask = sample_mask(bs=self.config.batch_size, nb_genes=self.x_dim, m_low=self.m_low, m_high=self.m_high)

        mask, b = mask

        # b = sample_mask_tf(bs=bs, nb_genes=self.x_dim, m_low=self.m_low, m_high=self.m_high)
        # b = sample_mask(bs=self.config.batch_size, nb_genes=self.x_dim, m_low=0.5, m_high=0.5)
        hint = b * mask + 0.5 * (1 - b)

        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generator forward pass
            x_gen = self.call((x * mask, cat, num, mask), training=True)

            # Compute losses
            sup_loss = self.config['lambd_sup'] * self.supervised_loss(x, x_gen, mask)
            gen_loss = sup_loss

        gen_grad = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grad, self.gen.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_gen)
        # Return a dict mapping metric names to current value
        return {**{'gen_loss': gen_loss, 'sup_loss': sup_loss}, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        x, cat, num, mask = x

        bs = tf.shape(x)[0]
        # if not self.config.inplace_mode:
        #    mask = sample_mask_tf(bs=bs, nb_genes=self.x_dim, m_low=self.m_low, m_high=self.m_high)

        if self.config.inplace_mode:
            mask, b = mask
        else:
            b = sample_mask_tf(bs=bs, nb_genes=self.x_dim, m_low=0.5, m_high=0.5)
        hint = b * mask + 0.5 * (1 - b)

        # Generator forward pass
        x_gen = self.call((x, cat, num, mask), training=False)

        # Compute losses
        sup_loss = self.config['lambd_sup'] * self.supervised_loss(x, x_gen, mask)
        pred_loss = self.supervised_loss(x, x_gen, 1 - mask)
        gen_loss = sup_loss

        # Update the metrics.
        self.compiled_metrics.update_state(x, x_gen)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': pred_loss, 'gen_loss': gen_loss, 'sup_loss': sup_loss}, **{m.name: m.result() for m in self.metrics}}


