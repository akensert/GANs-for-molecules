import tensorflow as tf


class RelationalGraphConvLayer(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 num_bases=None,
                 activation='relu',
                 use_bias=True,
                 use_self_loop=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        super(RelationalGraphConvLayer, self).__init__(**kwargs)

        self.units = units
        self.num_bases = num_bases
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_self_loop = use_self_loop
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is None: return None
        return mask[1]

    def _init_W(self, shape, name=None):
        return self.add_weight(
            shape=shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True, name=name, dtype=tf.float32
        )

    def _init_b(self, shape, name=None):
        return self.add_weight(
            shape=shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True, name=name, dtype=tf.float32
        )

    def build(self, input_shape):

        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        if self.num_bases and self.num_bases < bond_dim:
            self.W_decomp = self._init_W(shape=(bond_dim, self.num_bases))
        else:
            self.num_bases = bond_dim

        self.W_neigh = self._init_W(shape=(self.num_bases, atom_dim, self.units))

        if self.use_bias:
            self.b_neigh = self._init_b(shape=(bond_dim, 1, self.units))

        if self.use_self_loop:
            self.W_self = self._init_W(shape=(atom_dim, self.units))

            if self.use_bias:
                self.b_self = self._init_b(shape=(1, self.units))

        self.built = True


    def call(self, inputs, mask=None, training=False):

        adjacency_tensor, feature_tensor = inputs

        x_neigh = tf.matmul(adjacency_tensor, feature_tensor[:, None, :, :])

        if hasattr(self, 'W_decomp'):
            W_neigh = tf.transpose(self.W_neigh, (1, 0, 2))
            W_neigh = tf.matmul(self.W_decomp, W_neigh)
            W_neigh = tf.transpose(W_neigh, (1, 0, 2))
        else:
            W_neigh = self.W_neigh

        x_neigh = tf.matmul(x_neigh, W_neigh)

        if hasattr(self, 'b_neigh'):
            x_neigh += self.b_neigh

        if mask:
            x_neigh *= tf.cast(mask[1][:, None, :, None], x_neigh.dtype)

        x_neigh = tf.reduce_mean(x_neigh, axis=1)

        if hasattr(self, 'W_self'):
            x_self = tf.matmul(feature_tensor, self.W_self)
            if hasattr(self, 'b_self'):
                x_self += self.b_self
            if mask:
                x_self *= tf.cast(mask[1][..., None], x_self.dtype)
            x_neigh += x_self

        return self.activation(x_neigh)


# Aliases
RGCLayer = RGCNLayer = RelationalGraphConvLayer
