import tensorflow as tf
import numpy as np

from scipy.spatial import distance

class GCIR3_v1(tf.keras.Model):
    tf.keras.backend.set_floatx('float64')
    def __init__(self):
        super(GCIR3_v1, self).__init__()

        # Regularizer, activation
        self.regularizer = tf.keras.regularizers.L1L2(l1=.0, l2=.001)
        self.act = tf.keras.layers.ELU()
        self.ln = tf.keras.layers.LayerNormalization()

        # Temporal embedding network, phi
        self.conv1 = tf.keras.layers.Conv2D(5, (1, 65), activation=None, kernel_regularizer=self.regularizer)
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.layers.Conv2D(10, (1, 30), activation=None, kernel_regularizer=self.regularizer)
        self.ln2 = tf.keras.layers.LayerNormalization()

        self.conv3 = tf.keras.layers.Conv2D(20, (1, 37), activation=None, kernel_regularizer=self.regularizer)
        self.ln3 = tf.keras.layers.LayerNormalization()

        # Agent that automatically selects disease-affected regions, pi
        self.agent = tf.keras.layers.Conv2D(1, (1, 20), activation=None, kernel_regularizer=self.regularizer)

        # Graph convolutional network that represents regional relations, psi
        self.gconv1 = tf.keras.layers.Conv2D(10, (1, 20), activation=None, kernel_regularizer=self.regularizer)
        self.gconv2 = tf.keras.layers.Conv2D(5, (1, 10), activation=None, kernel_regularizer=self.regularizer)

        # Densely connected layer that maps features to the decision nodes, rho
        self.dropout = tf.keras.layers.GaussianDropout(0.5)
        self.dense = tf.keras.layers.Dense(2, activation=None, kernel_regularizer=self.regularizer)

    def Laplace(self, x, index):
        x = tf.squeeze(x)
        laplace = np.empty(shape=(x.shape[0], x.shape[1], x.shape[1]))
        for batch in range(x.shape[0]):
            # Calculate all pairwise distances
            distv = distance.pdist(x[batch, :, :], metric="correlation")
            # Convert to a square symmetric distance matrix
            dist = distance.squareform(distv)
            sigma = np.mean(dist)
            # Get affinity from similarity matrix
            A_tilde = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            # Diagnostic region selection (node rejection)
            A_tilde = np.multiply(index[batch], A_tilde)
            A_tilde = np.multiply(index[batch], A_tilde.T)
            # Extract adjacency for analysis
            D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis=1)))
            D_tilde[np.isinf(D_tilde)] = 0.
            # print(D_tilde)
            L = D_tilde @ A_tilde @ D_tilde
            laplace[batch, :, :] = L
        return laplace

    def call(self, x, rl):
        # Region-wise feedforward network
        x = self.act(self.ln1(self.conv1(x)))
        x = self.act(self.ln2(self.conv2(x)))
        x = self.act(self.ln3(self.conv3(x)))

        x = tf.transpose(x, perm=[0, 1, 3, 2])

        # TODO: This framework can be improved by introducing another hard/soft-attention mechanism.
        # TODO: Further, the hard-attention mechanism need not be reinforcement learning-based.
        if rl == True:
            # Agent that automatically selects diagnostically informative ROIs
            index = tf.squeeze(tf.keras.activations.sigmoid(self.agent(x))) # Step activation, [Batch, 114]
        else:
            index = tf.ones([x.shape[0], 114], dtype=tf.dtypes.float64)

        # Graph convolutional network for regional relation representation instead of summation pooling
        # TODO: Spektral API can be used for simpler and clearer graph neural network implementation.
        L = self.Laplace(x, index) # Normalized Laplace matrix [Batch, 114, 114]
        x = tf.squeeze(x) # Input feature [Batch, 114, F3], Note that F3 is feature dimension, not temporal dimension
        index = tf.tile(tf.expand_dims(index, -1), (1, 1, x.shape[-1]))
        x = tf.math.multiply(x, index)
        graph = tf.expand_dims(x, axis=-1) # Spectral graph construction [Batch, 114, F3]

        hidden = self.gconv1(graph) # [Batch, 114, Fg1]
        hidden = L @ tf.squeeze(hidden)
        hidden = self.act(hidden)

        hidden = tf.expand_dims(hidden, axis=-1)
        hidden = self.gconv2(hidden)
        hidden = L @ tf.squeeze(hidden)
        hidden = self.act(hidden) # [Batch, 114, 5]

        # Readout pooling
        # TODO: Spektral API can be used (Ref. Spektral-GlobalAvgPool and GlobalMaxPool).
        hidden_pos_max = tf.squeeze(tf.math.reduce_max(hidden, axis=1))
        hidden_neg_max = 1/tf.squeeze(tf.math.reduce_min(1/hidden, axis=1))
        hidden_max = np.empty(shape=hidden_pos_max.shape)
        for i in range(hidden_pos_max.shape[0]):
            for j in range(hidden_pos_max.shape[-1]):
                hidden_max[i, j] = hidden_pos_max[i, j]
                if hidden_pos_max[i, j] == 0:
                    hidden_max[i, j] = hidden_neg_max[i, j]

        hidden_sum = tf.squeeze(tf.math.reduce_sum(hidden, axis=1)) # [Batch, Fg2]
        feature = tf.concat((hidden_max, hidden_sum), axis=1) # [Batch, 2 * Fg2]

        # Decision making
        feature = self.dropout(feature)
        return tf.keras.activations.softmax(self.dense(feature))