import tensorflow as tf
import tensorflow_probability as tfp
from models.base_imputer import BaseImputer
from models.train_utils import sample_mask_tf
from data.data_utils import sample_mask

tfk = tf.keras
tfkl = tf.keras.layers


def make_generator(x_dim, vocab_sizes, nb_numeric, z_dim, nb_layers, hdim, bn=True, dropout=0.):
    # Define inputs
    x = tfkl.Input((x_dim,))
    z = tfkl.Input((z_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)
    mask = tfkl.Input((x_dim,), dtype=tf.float32)

    embed_cats = []
    total_emb_dim = 0

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)
        total_emb_dim += emb_dim
    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])
    total_emb_dim += nb_numeric

    # Include GAIN information
    embeddings = tfkl.Concatenate(axis=-1)([x, embeddings, mask])
    total_emb_dim += 2 * x_dim

    gen_emb = make_generator_emb(x_dim=x_dim,
                                 emb_dim=total_emb_dim,
                                 z_dim=z_dim,
                                 nb_layers=nb_layers,
                                 hdim=hdim,
                                 bn=bn,
                                 dropout=dropout)
    model = tfk.Model(inputs=[x, z, cat, num, mask], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model


def make_generator_emb(x_dim, emb_dim, z_dim, nb_layers=2, hdim=256, bn=True, dropout=0.):
    z = tfkl.Input((z_dim,))
    t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
    h = tfkl.Concatenate(axis=-1)([z, t_emb])

    for _ in range(nb_layers):
        h = tfkl.Dense(hdim)(h)
        if bn:
            h = tfkl.BatchNormalization()(h)
        h = tfkl.ReLU()(h)
        if dropout > 0:
            h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(x_dim)(h)
    model = tfk.Model(inputs=[z, t_emb], outputs=h)
    return model


def make_discriminator(x_dim, vocab_sizes, nb_numeric, nb_layers=2, hdim=256, bn=True, dropout=0.):
    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)
    hint = tfkl.Input((x_dim,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings, hint])
    for _ in range(nb_layers):
        h = tfkl.Dense(hdim)(h)
        if bn:
            h = tfkl.BatchNormalization()(h)
        h = tfkl.ReLU()(h)
        if dropout > 0:
            h = tfkl.Dropout(dropout)(h)
        h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(x_dim)(h)
    model = tfk.Model(inputs=[x, cat, num, hint], outputs=h)
    # model.summary()
    return model


class GAINGTEx(BaseImputer):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, m_low=0.5, m_high=0.5, name='GAINGTEx', **kwargs):
        super(GAINGTEx, self).__init__(x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name=name,
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
        self.disc = make_discriminator(x_dim=self.x_dim,
                                  vocab_sizes=self.vocab_sizes,
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
        assert len(optimizer) == 2
        self.gen_opt, self.disc_opt = optimizer

    def discriminator_loss(self, mask, disc_output, b):
        # mask: Variables being masked. 0: Variable is masked as input of generator. 1: Variable is kept
        loss = tf.nn.sigmoid_cross_entropy_with_logits(mask, disc_output)
        loss = tf.reduce_sum(loss * (1 - b), axis=-1)  # Shape=(nb_samples,)
        b_counts = tf.reduce_sum(1 - b, axis=-1)  # Shape=(nb_samples,)
        return tf.reduce_mean(loss / (b_counts + 1))

    def supervised_loss(self, x, x_gen, mask, eps=1e-7):
        # Masks variables that were discarded as input in the forward pass
        x_ = x * mask  # Input variables
        x_gen_ = x_gen * mask  # Reconstructed input variables
        mask_counts = tf.reduce_sum(mask, axis=-1)  # Shape=(nb_samples, )
        loss = tf.reduce_sum(mask * tf.math.squared_difference(x_, x_gen_), axis=-1)

        return tf.reduce_mean(loss / (mask_counts + eps))

    def generator_loss(self, mask, disc_output, b):
        # mask: Variables being masked. 0: Variable is masked as input of generator. 1: Variable is kept
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(mask, disc_output)  # Shape=(nb_samples, nb_vars)
        # return tf.reduce_mean(loss)

        # Compute loss on masked elements only
        loss = (1 - mask) * tf.nn.sigmoid_cross_entropy_with_logits(1 - mask,
                                                                    disc_output)  # Shape=(nb_samples, nb_vars)
        loss = tf.reduce_sum(loss * (1 - b), axis=-1)  # Shape=(nb_samples, )
        b_counts = tf.reduce_sum(1 - b, axis=-1)  # Shape=(nb_samples, )
        return tf.reduce_mean(loss / (b_counts + 1))

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

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generator forward pass
            x_gen = self.call((x * mask, cat, num, mask), training=False)
            x_gen_ = x * mask + x_gen * (1 - mask)

            # Forward pass on discriminator
            disc_out = self.disc([x_gen_, cat, num, hint], training=True)

            # Compute losses
            disc_loss = self.discriminator_loss(mask, disc_out, b)

        disc_grad = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(disc_grad, self.disc.trainable_variables))

        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generator forward pass
            x_gen = self.call((x * mask, cat, num, mask), training=True)
            x_gen_ = x * mask + x_gen * (1 - mask)

            # Forward pass on discriminator
            disc_out = self.disc([x_gen_, cat, num, hint], training=False)

            # Compute losses
            sup_loss = self.config['lambd_sup'] * self.supervised_loss(x, x_gen, mask)
            gen_loss = self.generator_loss(mask, disc_out, b) + sup_loss

        gen_grad = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grad, self.gen.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_gen)
        # Return a dict mapping metric names to current value
        return {**{'gen_loss': gen_loss, 'sup_loss': sup_loss, 'disc_loss': disc_loss}, **{m.name: m.result() for m in self.metrics}}

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
        x_gen_ = x * mask + x_gen * (1 - mask)

        # Forward pass on discriminator
        disc_out = self.disc([x_gen_, cat, num, hint], training=False)

        # Compute losses
        sup_loss = self.config['lambd_sup'] * self.supervised_loss(x, x_gen, mask)
        gen_loss = self.generator_loss(mask, disc_out, b) + sup_loss
        disc_loss = self.discriminator_loss(mask, disc_out, b)
        pred_loss = self.supervised_loss(x, x_gen, 1 - mask)

        # Update the metrics.
        self.compiled_metrics.update_state(x, x_gen)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': pred_loss, 'gen_loss': gen_loss, 'sup_loss': sup_loss, 'disc_loss': disc_loss}, **{m.name: m.result() for m in self.metrics}}


