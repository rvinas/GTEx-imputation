import tensorflow as tf
import numpy as np
from utils import *
import os
import datetime
from tf_utils import limit_gpu
import argparse

tfk = tf.keras
tfkl = tf.keras.layers

EPOCHS = 10000
BATCH_SIZE = 32
LATENT_DIM = 256
TISSUE_EMBEDDING_DIM = 8
CHECKPOINTS_DIR = '../checkpoints/'
MODELS_DIR = 'checkpoints/models/'
CORRECTED = False


def make_generator(x_dim, vocab_sizes, nb_numeric, z_dim=LATENT_DIM, bn=True, dropout=0.):
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
                                 bn=bn,
                                 dropout=dropout)
    model = tfk.Model(inputs=[x, z, cat, num, mask], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model, gen_emb


def make_generator_emb(x_dim, emb_dim, z_dim=LATENT_DIM, bn=True, dropout=0.):
    z = tfkl.Input((z_dim,))
    t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
    h = tfkl.Concatenate(axis=-1)([z, t_emb])
    h = tfkl.Dense(256)(h)
    if bn:
        h = tfkl.BatchNormalization()(h)
    h = tfkl.ReLU()(h)
    if dropout > 0:
        h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(256)(h)
    if bn:
        h = tfkl.BatchNormalization()(h)
    h = tfkl.ReLU()(h)
    if dropout > 0:
        h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(x_dim)(h)
    model = tfk.Model(inputs=[z, t_emb], outputs=h)
    return model


def make_discriminator(x_dim, vocab_sizes, nb_numeric, bn=True, dropout=0.):
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
    h = tfkl.Dense(256)(h)
    if bn:
        h = tfkl.BatchNormalization()(h)
    h = tfkl.ReLU()(h)
    if dropout > 0:
        h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(256)(h)
    if bn:
        h = tfkl.BatchNormalization()(h)
    h = tfkl.ReLU()(h)
    if dropout > 0:
        h = tfkl.Dropout(dropout)(h)
    h = tfkl.Dense(x_dim)(h)
    model = tfk.Model(inputs=[x, cat, num, hint], outputs=h)
    # model.summary()
    return model


def supervised_loss(x, x_gen, mask, eps=1e-7):
    # Masks variables that were discarded as input in the forward pass
    x_ = x * mask  # Input variables
    x_gen_ = x_gen * mask  # Reconstructed input variables
    mask_counts = tf.reduce_sum(mask, axis=-1)  # Shape=(nb_samples, )
    loss = tf.reduce_sum(mask * tf.math.squared_difference(x_, x_gen_), axis=-1)

    return tf.reduce_mean(loss / (mask_counts + eps))


def generator_loss(mask, disc_output, b):
    # mask: Variables being masked. 0: Variable is masked as input of generator. 1: Variable is kept
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(mask, disc_output)  # Shape=(nb_samples, nb_vars)
    # return tf.reduce_mean(loss)

    # Compute loss on masked elements only
    loss = (1 - mask) * tf.nn.sigmoid_cross_entropy_with_logits(1 - mask, disc_output)  # Shape=(nb_samples, nb_vars)
    loss = tf.reduce_sum(loss * (1 - b), axis=-1)  # Shape=(nb_samples, )
    b_counts = tf.reduce_sum(1 - b, axis=-1)  # Shape=(nb_samples, )
    return tf.reduce_mean(loss / (b_counts + 1))


def discriminator_loss(mask, disc_output, b):
    # mask: Variables being masked. 0: Variable is masked as input of generator. 1: Variable is kept
    loss = tf.nn.sigmoid_cross_entropy_with_logits(mask, disc_output)
    loss = tf.reduce_sum(loss * (1 - b), axis=-1)  # Shape=(nb_samples,)
    b_counts = tf.reduce_sum(1 - b, axis=-1)  # Shape=(nb_samples,)
    return tf.reduce_mean(loss / (b_counts + 1))


@tf.function
def train_disc(x, z, cc, nc, mask, hint, b, gen, disc, disc_opt):
    x_ = x * mask
    z_ = z * (1 - mask)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass
        x_gen = gen([x_, z_, cc, nc, mask], training=False)
        x_gen_ = x_ + x_gen * (1 - mask)

        # Forward pass on discriminator
        disc_out = disc([x_gen_, cc, nc, hint], training=True)

        # Compute losses
        disc_loss = discriminator_loss(mask, disc_out, b)

    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return disc_loss


@tf.function
def train_gen(x, z, cc, nc, mask, hint, b, gen, disc, gen_opt, lambd_sup=1):
    x_ = x * mask
    z_ = z * (1 - mask)

    with tf.GradientTape() as gen_tape:
        # Generator forward pass
        x_gen = gen([x_, z_, cc, nc, mask], training=True)
        x_gen_ = x_ + x_gen * (1 - mask)

        # Forward pass on discriminator
        disc_out = disc([x_gen_, cc, nc, hint], training=False)

        # Compute losses
        sup_loss = lambd_sup * supervised_loss(x, x_gen, mask)
        gen_loss = generator_loss(mask, disc_out, b) + sup_loss

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))

    return gen_loss, sup_loss


def get_mask_hint_b(bs, nb_genes, m_low=0.5, m_high=0.95, b_low=0.5, b_high=0.5):
    # Compute masks
    # m_low = 0.5?
    p_mask = np.random.uniform(low=m_low, high=m_high, size=(bs,))  # Probability of setting mask to 0
    mask = np.random.binomial(1, p_mask, size=(nb_genes, bs)).astype(np.float32).T  # Shape=(bs, nb_genes)

    # Compute hint
    p_b = np.random.uniform(low=b_low, high=b_high, size=(bs,))
    b = np.random.binomial(1, p_b, size=(nb_genes, bs)).T  # Shape=(bs, nb_genes)
    hint = b * mask + 0.5 * (1 - b)

    return mask, hint, b.astype(np.float32)


def train(dataset, cat_covs, num_covs, epochs, batch_size, gen, disc, gen_opt, disc_opt, score_fn, save_fn, z_dim=None,
          verbose=True, checkpoint_dir='./checkpoints/cpkt', log_dir='./logs/', patience=20, lambd_sup=1, b_low=0.5, b_high=0.5):
    nb_samples, nb_genes = dataset.shape
    if z_dim is None:
        z_dim = nb_genes

    # Set up logs and checkpoints
    """
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    gen_log_dir = log_dir + current_time + '/gen_gain'
    disc_log_dir = log_dir + current_time + '/disc_gain'
    gamma_dxdz_dir = log_dir + current_time + '/gamma_dxdz_gain'
    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
    """
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    score_dir = log_dir + current_time + '/mse_gain'
    score_writer = tf.summary.create_file_writer(score_dir)


    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    gen_losses = tfk.metrics.Mean('gen_loss', dtype=tf.float32)
    sup_losses = tfk.metrics.Mean('sup_loss', dtype=tf.float32)
    disc_losses = tfk.metrics.Mean('disc_loss', dtype=tf.float32)
    best_score = -np.inf
    initial_patience = patience

    for epoch in range(epochs):
        shuffled_idxs = np.arange(nb_samples)
        np.random.shuffle(shuffled_idxs)
        dataset_ = dataset[shuffled_idxs]
        cat_covs_ = cat_covs[shuffled_idxs]
        num_covs_ = num_covs[shuffled_idxs]

        for i in range(0, nb_samples, batch_size):
            x = dataset_[i: i + batch_size, :]
            cc = cat_covs_[i: i + batch_size, :]
            nc = num_covs_[i: i + batch_size, :]
            bs = x.shape[0]

            mask, hint, b = get_mask_hint_b(bs, nb_genes, b_low=b_low, b_high=b_high)
            z = tf.random.normal([bs, z_dim])

            disc_loss = train_disc(x, z, cc, nc, mask, hint, b, gen, disc, disc_opt)
            gen_loss, sup_loss = train_gen(x, z, cc, nc, mask, hint, b, gen, disc, gen_opt, lambd_sup=lambd_sup)
            # gen_loss, sup_loss, disc_loss = train_step(x, z, cc, nc, mask, hint, b, gen, disc, gen_opt, disc_opt)
            disc_losses(disc_loss)
            sup_losses(sup_loss)
            gen_losses(gen_loss)

        # Logs
        """
        with disc_summary_writer.as_default():
            tf.summary.scalar('disc_loss', disc_losses.result(), step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar('gen_loss', disc_losses.result(), step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar('sup_loss', gen_losses.result(), step=epoch)
        """

        # Save the model
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            score = score_fn(gen)
            with score_writer.as_default():
                tf.summary.scalar('score', -score, step=epoch)

            if score > best_score:
                print('Saving model ...')
                save_fn()
                best_score = score
                patience = initial_patience
            else:
                patience -= 1

            if verbose:
                print('Score: {:.3f}'.format(score))

            if verbose:
                print('Epoch {}. Gen loss: {:.2f}. Sup loss: {:.2f}. Disc loss: {:.2f}'.format(epoch + 1,
                                                                                               gen_losses.result(),
                                                                                               sup_losses.result(),
                                                                                               disc_losses.result()))
        gen_losses.reset_states()
        sup_losses.reset_states()
        disc_losses.reset_states()

        if patience == 0:
            break


def predict(x, cc, nc, mask, gen, z=None, training=False):
    nb_samples = cc.shape[0]
    if z is None:
        z_dim = gen.input[0].shape[-1]
        z = tf.random.normal([nb_samples, z_dim])
    x_ = x * mask
    z_ = z * (1 - mask)
    out = gen([x_, z_, cc, nc, mask], training=training)
    if not training:
        return out.numpy()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd_sup', dest='lambd_sup', nargs='?', default=1., type=float)
    parser.add_argument('--dropout', dest='dropout', nargs='?', default=0, type=float)
    parser.add_argument('--bn', dest='bn', action='store_true', default=False)
    # parser.add_argument('--m_low', dest='m_low', nargs='?', default=0.5, type=float)
    # parser.add_argument('--m_high', dest='m_high', nargs='?', default=0.95, type=float)
    parser.add_argument('--b_low', dest='b_low', nargs='?', default=0.5, type=float)
    parser.add_argument('--b_high', dest='b_high', nargs='?', default=0.5, type=float)
    parser.add_argument('--epochs', dest='epochs', nargs='?', default=10000, type=int)

    # parser.add_argument('--resume_training', dest='resume_training', nargs='?', default=False, type=str)
    args = parser.parse_args()

    # GPU limit
    limit_gpu()

    mask, hint, b = get_mask_hint_b(32, 100)
    print('Mean mask: ', np.mean(mask))
    print('Mean b: ', np.mean(b))

    # Load dataset
    x, symbols, sampl_ids, tissues = load_gtex(corrected=CORRECTED)
    x = standardize(x)
    x = np.float32(x)

    # Load metadata
    df_metadata = gtex_metadata()
    print(df_metadata.head())

    # Process categorical metadata
    cat_cols = ['SEX', 'COHORT']  # 'SEX', 'COHORT'
    df_metadata[cat_cols] = df_metadata[cat_cols].astype('category')
    # cat_map = df_metadata[cat_cols].cat.categories
    # print('cat map', cat_map)
    cat_dicts = [df_metadata[cat_col].cat.categories.values for cat_col in cat_cols]
    df_metadata[cat_cols] = df_metadata[cat_cols].apply(lambda x: x.cat.codes)
    cat_covs = df_metadata.loc[sampl_ids, cat_cols].values
    tissues_dict_inv = np.array(list(sorted(set(tissues))))
    tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
    tissues = np.vectorize(lambda t: tissues_dict[t])(tissues)
    cat_dicts.append(tissues_dict_inv)
    cat_covs = np.concatenate((cat_covs, tissues[:, None]), axis=-1)
    cat_covs = np.int32(cat_covs)
    print('Cat covs: ', cat_covs.shape)

    # Process numerical metadata
    num_cols = ['AGE']  # 'AGE'
    num_covs = df_metadata.loc[sampl_ids, num_cols].values
    num_covs = standardize(num_covs)
    num_covs = np.float32(num_covs)
    # num_covs = np.zeros_like(cat_covs).astype(np.float32)

    # Train/test split
    x_train, x_test, sampl_ids_train, sampl_ids_test = split_train_test_v2(x, sampl_ids)
    print(sampl_ids_test)
    num_covs_train, num_covs_test, _, _ = split_train_test_v2(num_covs, sampl_ids)
    cat_covs_train, cat_covs_test, _, _ = split_train_test_v2(cat_covs, sampl_ids)
    x_train, x_val, _, sampl_ids_val = split_train_test_v2(x_train, sampl_ids_train, train_rate=0.8)
    num_covs_train, num_covs_val, _, _ = split_train_test_v2(num_covs_train, sampl_ids_train, train_rate=0.8)
    cat_covs_train, cat_covs_val, sampl_ids_train, sampl_ids_val = split_train_test_v2(cat_covs_train, sampl_ids_train, train_rate=0.8)


    # Define model
    vocab_sizes = [len(c) for c in cat_dicts]
    print('Vocab sizes: ', vocab_sizes)
    nb_numeric = num_covs.shape[-1]
    x_dim = x.shape[-1]
    z_dim = x_dim
    gen, gen_emb = make_generator(x_dim=x_dim,
                                  vocab_sizes=vocab_sizes,
                                  nb_numeric=nb_numeric,
                                  z_dim=z_dim,
                                  bn=args.bn,
                                  dropout=args.dropout)
    disc = make_discriminator(x_dim=x_dim,
                              vocab_sizes=vocab_sizes,
                              nb_numeric=nb_numeric,
                              bn=args.bn,
                              dropout=args.dropout)


    # Evaluation metrics
    def score_fn(x_val, cat_covs_val, num_covs_val):
        def _score(gen):
            bs, nb_genes = x_val.shape
            mask, _, _ = get_mask_hint_b(bs, nb_genes, m_low=0.5)
            x_gen = predict(x=x_val,
                            cc=cat_covs_val,
                            nc=num_covs_val,
                            mask=mask,
                            gen=gen)
            imp_mse = np.sum((1 - mask) * (x_gen - x_val) ** 2) / np.sum(1 - mask)
            # Compute upper bound with x_train?
            return -imp_mse

        return _score


    # Function to save models
    def save_fn(models_dir=MODELS_DIR):
        name = 'gen_gain_l{}.h5'.format(args.lambd_sup)
        if CORRECTED:
            name = 'gen_gain_l{}_CORRECTED.h5'.format(args.lambd_sup)
        try:
            gen.save(models_dir + name)
            save_fn.n = 0
        except OSError:
            print('ERROR: Could not save model')
            save_fn.n += 1


    # Train model
    gen_opt = tfk.optimizers.RMSprop(1e-3)
    disc_opt = tfk.optimizers.RMSprop(1e-3)

    train(dataset=x_train,
          cat_covs=cat_covs_train,
          num_covs=num_covs_train,
          z_dim=z_dim,
          batch_size=BATCH_SIZE,
          epochs=args.epochs,
          gen=gen,
          disc=disc,
          gen_opt=gen_opt,
          disc_opt=disc_opt,
          lambd_sup=args.lambd_sup,
          b_low=args.b_low,
          b_high=args.b_high,
          score_fn=score_fn(x_val, cat_covs_val, num_covs_val),
          save_fn=save_fn)

    # Evaluate data
    score = score_fn(x_test, cat_covs_test, num_covs_test)(gen)
    print('Score: {:.2f}'.format(score))
