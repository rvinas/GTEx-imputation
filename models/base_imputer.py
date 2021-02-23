import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers



class BaseImputer(tfk.Model):
    def __init__(self, x_dim, vocab_sizes, nb_numeric, nb_categoric, config, name='BaseImputer', **kwargs):
        super(BaseImputer, self).__init__(name=name, **kwargs)
        self.x_dim = x_dim
        self.vocab_sizes = vocab_sizes
        self.nb_numeric = nb_numeric
        self.nb_categoric = nb_categoric
        self.config = config
        self.model = self._create_model()

    def _create_model(self):
        # Define inputs
        x = tfkl.Input((self.x_dim,))
        nb_categoric = len(self.vocab_sizes)
        cat = tfkl.Input((self.nb_categoric,), dtype=tf.int32)
        num = tfkl.Input((self.nb_numeric,), dtype=tf.float32)
        mask = tfkl.Input((self.x_dim,), dtype=tf.float32)

        embed_cats = []
        for n, vs in enumerate(self.vocab_sizes):
            emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
            c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                                   output_dim=emb_dim  # Embedding size
                                   )(cat[:, n])
            embed_cats.append(c_emb)
        if nb_categoric == 1:
            embeddings = embed_cats[0]
        else:
            embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
        embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])

        # Include information
        embeddings = tfkl.Concatenate(axis=-1)([x, embeddings, mask])
        h = embeddings
        for _ in range(self.config['nb_layers']):
            h = tfkl.Dense(self.config['hdim'])(h)
            if self.config['bn']:
                h = tfkl.BatchNormalization()(h)
            h = tfkl.ReLU()(h)
            if self.config['dropout'] > 0:
                h = tfkl.Dropout(self.config['dropout'])(h)
        h = tfkl.Dense(self.x_dim)(h)

        model = tfk.Model(inputs=[x, cat, num, mask], outputs=h)
        model.summary()
        return model

    def loss_fn(self, x, x_gen, eps=1e-7):
        raise NotImplementedError('Please implement method loss_fn when subclassing BaseImputer')


    def compile(self, optimizer):
        super(BaseImputer, self).compile()
        self.optimizer = optimizer

    def call(self, x, **kwargs):
        x, cat, num, mask = x
        x_ = x * mask
        return self.model([x_, cat, num, mask], **kwargs)

    def impute(self, x, cat, num, mask, **kwargs):
        x_ = x * mask
        x_imp = self.model([x_, cat, num, mask], **kwargs)
        x_imp = x_ + x_imp * (1 - mask)
        return x_imp

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x, cat, num, mask = x

        with tf.GradientTape() as tape:
            y_pred = self.call((x, cat, num, mask), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss_fn((x, mask), y_pred)  # compiled_loss((x, mask), y_pred, regularization_losses=self.losses)

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

        # Compute predictions
        y_pred = self.call((x, cat, num, mask), training=False)
        # Updates the metrics tracking the loss
        loss = self.loss_fn((x, mask), y_pred)  # compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}
