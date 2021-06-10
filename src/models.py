import tensorflow as tf
import numpy as np

from layers import RelationalGraphConvLayer


class GraphWGAN(tf.keras.Model):

    def __init__(self,
                 generator,
                 discriminator,
                 latent_dim=64,
                 discriminator_steps=1,
                 generator_steps=1,
                 gp_weight=10,
                 **kwargs):

        super(GraphWGAN, self).__init__(**kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.gp_weight = gp_weight

    def compile(self, optimizer_generator, optimizer_discriminator):
        super(GraphWGAN, self).compile()
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.metric_generator = tf.keras.metrics.Mean(name="loss_gen")
        self.metric_discriminator = tf.keras.metrics.Mean(name="loss_disc")

    def _loss_discriminator(self, graph_real, graph_gen):
        logits_real = self.discriminator(graph_real, training=True)
        logits_gen = self.discriminator(graph_gen, training=True)
        loss_real = tf.reduce_mean(logits_real)
        loss_gen = tf.reduce_mean(logits_gen)
        disc_loss = loss_gen - loss_real
        disc_loss += self._gradient_penalty(
            graph_real, graph_gen) * self.gp_weight
        return disc_loss

    def _loss_generator(self, graph_gen):
        logits_gen = self.discriminator(graph_gen, training=True)
        gen_loss = tf.reduce_mean(logits_gen)
        return -gen_loss

    def _train_step_discriminator(self, inputs):
        z = tf.random.normal((self.batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            decoded = self.generator(z, training=False)
            disc_loss = self._loss_discriminator(inputs, decoded)

        grads = tape.gradient(
            disc_loss, self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))
        self.metric_discriminator.update_state(disc_loss)

    def _train_step_generator(self, inputs):
        # inputs not passed in GAN or WGAN
        z = tf.random.normal((self.batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            decoded = self.generator(z, training=True)
            gen_loss = self._loss_generator(decoded)

        grads = tape.gradient(
            gen_loss, self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        self.metric_generator.update_state(gen_loss)

    def train_step(self, inputs):
        # add random label noise? 0.05 * uniform(labels.shape)
        x = inputs
        self.batch_size = tf.shape(x[0])[0]

        # discriminator training step(s)
        for _ in range(self.discriminator_steps):
            self._train_step_discriminator(x)

        # generator & encoder training step(s)
        for _ in range(self.generator_steps):
            self._train_step_generator(x)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        # if call method is not implemented,
        # model.fit(generator) will raise error
        return inputs

    def _gradient_penalty(self, graph_real, graph_gen):

        adj_real, feat_real = graph_real
        adj_gen, feat_gen = graph_gen

        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adj_interp = (adj_real * alpha) + (1-alpha) * adj_gen

        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        feat_interp = (feat_real * alpha) + (1-alpha) * feat_gen

        with tf.GradientTape() as tape:
            tape.watch(adj_interp)
            tape.watch(feat_interp)
            logits = self.discriminator([adj_interp, feat_interp],
                                        training=True)

        grads = tape.gradient(logits, [adj_interp, feat_interp])

        grads_adj_penalty = (1 - tf.norm(grads[0], axis=1))**2
        grads_feat_penalty = (1 - tf.norm(grads[1], axis=2))**2

        return tf.reduce_mean(
            tf.reduce_mean(grads_adj_penalty, axis=(-2, -1)) +
            tf.reduce_mean(grads_feat_penalty, axis=(-1))
        )

    def generate(self, batch_size=1):
        z = tf.random.normal((batch_size, self.latent_dim))
        A, X = self.generator(z, training=False, discretize=True)
        return tf.squeeze(A).numpy(), tf.squeeze(X).numpy()

    def discriminate(self, inputs):
        A, X = inputs
        if tf.rank(A) == 3: A = tf.expand_dims(A, 0)
        if tf.rank(X) == 2: X = tf.expand_dims(X, 0)
        logits = self.discriminator([A, X], training=False)
        return tf.nn.sigmoid(tf.squeeze(logits)).numpy()


class GraphGAN(GraphWGAN):

    def __init__(self,
                 generator,
                 discriminator,
                 latent_dim=64,
                 discriminator_steps=1,
                 generator_steps=1,
                 gp_weight=None,
                 **kwargs):
        super(GraphGAN, self).__init__(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            gp_weight=gp_weight,
            **kwargs)

    def _loss_discriminator(self, graph_real, graph_gen):
        logits_real = self.discriminator(graph_real, training=True)
        logits_gen = self.discriminator(graph_gen, training=True)
        return (
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(logits_gen), logits_gen) +
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(logits_real), logits_real)
        )

    def _loss_generator(self, graph_gen):
        logits_gen = self.discriminator(graph_gen, training=False)
        return (
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(logits_gen), logits_gen)
        )


class GraphVAEGAN(GraphGAN):

    def __init__(self,
                 encoder,
                 generator,
                 discriminator,
                 discriminator_steps=1,
                 generator_steps=1,
                 rec_weight_enc=1.0,
                 rec_weight_gen=1e-2,
                 kl_weight=1e-3,
                 gp_weight=None,
                 **kwargs):

        super(GraphVAEGAN, self).__init__(
            generator=generator,
            discriminator=discriminator,
            latent_dim=encoder.out_shape[0],
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            gp_weight=gp_weight,
            **kwargs)

        self.encoder = encoder
        self.rec_weight_enc = rec_weight_enc
        self.rec_weight_gen = rec_weight_gen
        self.kl_weight = kl_weight

    def compile(self,
                optimizer_encoder,
                optimizer_generator,
                optimizer_discriminator):
        super(GraphVAEGAN, self).compile(
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator)
        self.optimizer_encoder = optimizer_encoder
        self.metric_encoder = tf.keras.metrics.Mean(name="loss_enc")

    @staticmethod
    def kl_divergence(z_mean, z_log_var):
        loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    @staticmethod
    def reconstruction_loss(x_real, x_generated):
        x_real = tf.math.l2_normalize(x_real, axis=2)
        x_generated = tf.math.l2_normalize(x_generated, axis=2)
        expanded_x_real = tf.expand_dims(x_real, 2)
        expanded_x_generated = tf.expand_dims(x_generated, 1)
        distances = tf.reduce_sum(
            tf.math.squared_difference(
                expanded_x_real, expanded_x_generated),
            axis=3)
        return tf.reduce_mean(tf.reduce_min(distances, axis=2))

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * .5) + mean

    def _loss_discriminator(self, graph_real, graph_enc, graph_ran):
        logits_enc  = self.discriminator(graph_enc, training=True)
        logits_ran  = self.discriminator(graph_ran, training=True)
        logits_real = self.discriminator(graph_real, training=True)
        return (
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(logits_enc), logits_enc) +
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(logits_ran), logits_ran) +
            tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(logits_real), logits_real)
        )

    def _loss_generator(self, graph_real, graph_enc, graph_ran, z_enc):

        feat_enc, logits_enc = self.discriminator(
            graph_enc, return_features=True, training=True)

        logits_ran = self.discriminator(
            graph_ran, training=True)

        feat_real, logits_real = self.discriminator(
            graph_real, return_features=True, training=True)

        feat_real = tf.concat(feat_real, axis=-1)
        feat_enc = tf.concat(feat_enc, axis=-1)
        rec_loss = self.reconstruction_loss(feat_real, feat_enc)
        #rec_loss = 0.
        #for freal, fenc in zip(feat_real, feat_enc):
        #    rec_loss += self.reconstruction_loss(freal, fenc)

        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(logits_enc), logits_enc)
        gen_loss += tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(logits_ran), logits_ran)
        gen_loss += rec_loss * self.rec_weight_gen

        enc_loss = self.kl_divergence(z_enc[..., 0], z_enc[..., 1]) * self.kl_weight
        enc_loss += rec_loss * self.rec_weight_enc

        return enc_loss, gen_loss

    def _train_step_discriminator(self, inputs):

        z_ran = tf.random.normal((self.batch_size, self.latent_dim))
        z_enc = self.encoder(inputs, training=True)
        z_enc = self.reparameterize(z_enc[...,0], z_enc[...,1])

        with tf.GradientTape() as tape:
            decoded_enc = self.generator(z_enc, training=True)
            decoded_ran = self.generator(z_ran, training=True)
            disc_loss = self._loss_discriminator(inputs, decoded_enc, decoded_ran)

        grads = tape.gradient(
            disc_loss, self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))
        self.metric_discriminator.update_state(disc_loss)

    def _train_step_generator(self, inputs):

        z_ran = tf.random.normal((self.batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape:

            z_enc = self.encoder(inputs, training=True)
            z_enc_reparm = self.reparameterize(z_enc[..., 0], z_enc[..., 1])

            decoded_enc = self.generator(z_enc_reparm, training=True)
            decoded_ran = self.generator(z_ran, training=True)

            enc_loss, gen_loss = self._loss_generator(
                inputs, decoded_enc, decoded_ran, z_enc)

        grads = gen_tape.gradient(
            gen_loss, self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        grads = enc_tape.gradient(
            enc_loss, self.encoder.trainable_weights)
        self.optimizer_encoder.apply_gradients(
            zip(grads, self.encoder.trainable_weights))

        self.metric_generator.update_state(gen_loss)
        self.metric_encoder.update_state(enc_loss)

    def encode(self, inputs, epsilon=False):
        A, X = inputs
        if tf.rank(A) == 3: A = tf.expand_dims(A, 0)
        if tf.rank(X) == 2: X = tf.expand_dims(X, 0)
        z = self.encoder([A, X], training=False)
        if epsilon:
            z = self.reparameterize(z[..., 0], z[..., 1])
        else:
            z = z[..., 0] + tf.exp(z[..., 1] * .5)
        return tf.squeeze(z).numpy()

    def decode(self, inputs):
        if tf.rank(inputs) == 1:
            inputs = tf.expand_dims(inputs, 0)
        graph = self.generator(inputs, discretize=True, training=False)
        return tf.squeeze(graph[0]).numpy(), tf.squeeze(graph[1]).numpy()


class GraphVAEWGAN(GraphVAEGAN):

    def __init__(self,
                 encoder,
                 generator,
                 discriminator,
                 discriminator_steps=1,
                 generator_steps=1,
                 gp_weight=10.0,
                 rec_weight_enc=10.0,
                 rec_weight_gen=1.0,
                 kl_weight=1e-4,
                 **kwargs):

        super(GraphVAEWGAN, self).__init__(
            generator=generator,
            discriminator=discriminator,
            discriminator_steps=discriminator_steps,
            generator_steps=generator_steps,
            encoder=encoder,
            gp_weight=gp_weight,
            rec_weight_enc=rec_weight_enc,
            rec_weight_gen=rec_weight_gen,
            kl_weight=kl_weight,
            **kwargs)

    def compile(self,
                optimizer_encoder,
                optimizer_generator,
                optimizer_discriminator):
        super(GraphVAEWGAN, self).compile(
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            optimizer_encoder=optimizer_encoder)

    def _loss_discriminator(self, graph_real, graph_enc, graph_ran):
        logits_enc  = self.discriminator(graph_enc, training=True)
        logits_ran  = self.discriminator(graph_ran, training=True)
        logits_real = self.discriminator(graph_real, training=True)
        loss_real = tf.reduce_mean(logits_real)
        loss_enc = tf.reduce_mean(logits_enc)
        loss_ran = tf.reduce_mean(logits_ran)
        disc_loss = ((loss_enc + loss_ran) / 2) - loss_real
        disc_loss += GraphWGAN._gradient_penalty(self, graph_real, graph_enc) * self.gp_weight * 0.5
        disc_loss += GraphWGAN._gradient_penalty(self, graph_real, graph_ran) * self.gp_weight * 0.5
        return disc_loss

    def _loss_generator(self, graph_real, graph_enc, graph_ran, z_enc):

        feat_enc, logits_enc = self.discriminator(
            graph_enc, return_features=True, training=False)

        logits_ran = self.discriminator(
            graph_ran, training=False)

        feat_real, logits_real = self.discriminator(
            graph_real, return_features=True, training=False)

        rec_loss = 0.
        for freal, fenc in zip(feat_real, feat_enc):
            rec_loss += self.reconstruction_loss(freal, fenc)

        gen_loss = -tf.reduce_mean(logits_enc)
        gen_loss -= tf.reduce_mean(logits_ran)
        gen_loss += rec_loss * self.rec_weight_gen

        enc_loss = self.kl_divergence(z_enc[..., 0], z_enc[..., 1]) * self.kl_weight
        enc_loss += rec_loss * self.rec_weight_enc

        return enc_loss, gen_loss

    def _train_step_discriminator(self, inputs):

        z_ran = tf.random.normal((self.batch_size, self.latent_dim))
        z_enc = self.encoder(inputs, training=True)
        z_enc = self.reparameterize(z_enc[...,0], z_enc[...,1])

        with tf.GradientTape() as tape:
            decoded_enc = self.generator(z_enc, training=True)
            decoded_ran = self.generator(z_ran, training=True)
            disc_loss = self._loss_discriminator(inputs, decoded_enc, decoded_ran)

        grads = tape.gradient(
            disc_loss, self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))
        self.metric_discriminator.update_state(disc_loss)

    def _train_step_generator(self, inputs):

        z_ran = tf.random.normal((self.batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape:

            z_enc = self.encoder(inputs, training=True)
            z_enc_reparm = self.reparameterize(z_enc[..., 0], z_enc[..., 1])

            decoded_enc = self.generator(z_enc_reparm, training=True)
            decoded_ran = self.generator(z_ran, training=True)

            enc_loss, gen_loss = self._loss_generator(
                inputs, decoded_enc, decoded_ran, z_enc)

        grads = gen_tape.gradient(
            gen_loss, self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        grads = enc_tape.gradient(
            enc_loss, self.encoder.trainable_weights)
        self.optimizer_encoder.apply_gradients(
            zip(grads, self.encoder.trainable_weights))

        self.metric_generator.update_state(gen_loss)
        self.metric_encoder.update_state(enc_loss)


class GraphEncoder(tf.keras.Model):

    def __init__(self,
                 out_shape,
                 max_atoms,
                 atom_dim,
                 bond_dim,
                 gconv_units=[128, 128, 128, 128],
                 dense_units=[1024, 1024],
                 gconv_activation='relu',
                 dense_activation='relu',
                 gconv_dropout_rate=0.0,
                 dense_dropout_rate=0.2,
                 gconv_kernel_initializer='glorot_uniform',
                 gconv_bias_initializer='zeros',
                 dense_kernel_initializer='glorot_uniform',
                 dense_bias_initializer='zeros',
                 gconv_kernel_regularizer=None,
                 dense_kernel_regularizer=None,
                 gconv_bias_regularizer=None,
                 dense_bias_regularizer=None,
                 gconv_use_bias=False,
                 dense_use_bias=True,
                 gconv_use_self_loop=False,
                 **kwargs):

        super(GraphEncoder, self).__init__(**kwargs)

        self.masking = tf.keras.layers.Masking(mask_value=0)
        self.gconv_dropout = tf.keras.layers.Dropout(gconv_dropout_rate)
        self.gconv_layers = [
            RelationalGraphConvLayer(
                 units=units,
                 num_bases=None,
                 activation=gconv_activation,
                 use_bias=gconv_use_bias,
                 use_self_loop=gconv_use_self_loop,
                 kernel_initializer=gconv_kernel_initializer,
                 bias_initializer=gconv_bias_initializer,
                 kernel_regularizer=gconv_kernel_regularizer,
                 bias_regularizer=gconv_bias_regularizer,
            )
            for units in gconv_units
        ]

        self.feat_shape = (-1, max_atoms, atom_dim)
        self.adj_shape = (-1, bond_dim, max_atoms, max_atoms)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense_dropout = tf.keras.layers.Dropout(dense_dropout_rate)
        self.dense_layers = [
            tf.keras.layers.Dense(
                units=units,
                activation=dense_activation,
                kernel_initializer=dense_kernel_initializer,
                bias_initializer=dense_bias_initializer,
                kernel_regularizer=dense_kernel_regularizer,
                bias_regularizer=dense_bias_regularizer,
                use_bias=dense_use_bias,
            )
            for units in dense_units
        ]

        if not isinstance(out_shape, (list, tuple)):
            self.out_shape = [out_shape]
        else:
            self.out_shape = out_shape

        self.dense_output = tf.keras.layers.Dense(
            units=tf.math.reduce_prod(self.out_shape), dtype='float32')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, return_features=None):

        adjacency_tensor, feature_tensor = inputs
        features_list = [feature_tensor]

        # masking used, although not needed as non-atom/bond is encoded
        feature_tensor = self.masking(feature_tensor)

        feature_tensor = tf.reshape(feature_tensor, self.feat_shape)
        adjacency_tensor = tf.reshape(adjacency_tensor, self.adj_shape)
        for i in range(len(self.gconv_layers)):
            feature_tensor = self.gconv_layers[i](
                inputs=[adjacency_tensor, feature_tensor],
                training=training
            )
            feature_tensor = self.gconv_dropout(feature_tensor)
            features_list.append(feature_tensor)

        x = self.pooling(feature_tensor)

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.dense_dropout(x, training=training)

        x = self.dense_output(x)
        x = tf.reshape(x, (-1, *self.out_shape))

        if return_features:
            return features_list, x
        return x


class GraphGenerator(tf.keras.Model):

    def __init__(self,
                 atom_shape=(30, 11),
                 bond_shape=(5, 30, 30),
                 units=(128, 256, 512),
                 activation='tanh',
                 dropout_rate=0.2,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        super(GraphGenerator, self).__init__(**kwargs)


        self.fc = MultiLayerPerceptron(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='mlp')

        self.adj_mapping = MultiLayerPerceptron(
            units=tf.math.reduce_prod(bond_shape),
            out_shape=(-1, *bond_shape),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='adj_map')

        self.feat_mapping = MultiLayerPerceptron(
            units=tf.math.reduce_prod(atom_shape),
            out_shape=(-1, *atom_shape),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='feat_map')


    def summary(self):
        return (
            self.fc.summary(),
            self.adj_mapping.summary(),
            self.feat_mapping.summary()
        )

    @staticmethod
    def symmetrify(inputs):
        return (inputs + tf.transpose(inputs, (0, 1, 3, 2))) / 2

    @tf.custom_gradient
    @staticmethod
    def discretize_adjacency(inputs):
        axis = 1
        def grad(upstream_gradient):
            return tf.identity(upstream_gradient)
        axis_shape = tf.shape(inputs)[axis]
        x = tf.argmax(inputs, axis=axis)
        x = tf.one_hot(x, depth=axis_shape, axis=axis)
        x = tf.linalg.set_diag(x, tf.zeros(tf.shape(x)[:-1]))
        return x, grad

    @tf.custom_gradient
    @staticmethod
    def discretize_features(inputs):
        axis = 2
        def grad(upstream_gradient):
            return tf.identity(upstream_gradient)
        axis_shape = tf.shape(inputs)[axis]
        x = tf.argmax(inputs, axis=axis)
        x = tf.one_hot(x, depth=axis_shape, axis=axis)
        return x, grad

    @tf.function
    def call(self, inputs, training=False, discretize=False):

        x = self.fc(inputs, training=training)

        x_adj = self.adj_mapping(x, training=training)

        x_adj = self.symmetrify(x_adj)
        x_adj = tf.nn.softmax(x_adj, axis=1)

        x_feat = self.feat_mapping(x, training=training)
        x_feat = tf.nn.softmax(x_feat, axis=-1)

        if discretize:
            x_adj = self.discretize_adjacency(x_adj)
            x_feat = self.discretize_features(x_feat)

        return x_adj, x_feat


class MultiLayerPerceptron(tf.keras.Model):

    def __init__(self,
                 units,
                 out_shape=None,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 dropout_rate=0.0,
                 **kwargs):

        super(MultiLayerPerceptron, self).__init__(**kwargs)

        if not isinstance(units, (list, tuple)):
            units = [units]

        self.out_shape = out_shape
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dense_layers = [
            tf.keras.layers.Dense(
                units=u,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            for u in units
        ]

    def call(self, inputs, mask=None, training=False):

        x = inputs

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.dropout(x)

        if self.out_shape is not None:
            x = tf.reshape(x, self.out_shape)

        return x


# Aliases
MLP = MultilayerPerceptron = MultiLayerPerceptron
Encoder = GraphConvEncoder = MoleculeEncoder = GraphEncoder
Decoder = GraphConvGenerator = MoleculeGenerator = GraphGenerator
