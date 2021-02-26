import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers


def sample_mask_tf(bs, nb_genes, m_low=0.5, m_high=0.5):
    # Compute masks
    # m_low = 0.5?
    p_mask = tf.random.uniform(shape=(bs,), minval=m_low, maxval=m_high)  # Probability of setting mask to 0
    binomial = tfp.distributions.Binomial(1, probs=p_mask)
    mask = tf.transpose(binomial.sample(nb_genes))
    return mask
