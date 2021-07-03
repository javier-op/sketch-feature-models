import tensorflow as tf

class VAE(tf.keras.Model):
    """
    Variational Auto Encoder wrapper, encoder and decoder architectures
    shoulde be instantiated before and passed in the constructor.
    https://arxiv.org/abs/1312.6114
    """
    
    def __init__(self, encoder, decoder, inner_shape, code_size):
        """
        Variatonal Auto Encoder constructor.
        Args:
            encoder (tensorflow model): encoder architecture
            decoder (tensorflow model): decoder architecture
            inner_shape (integer): size of encoder output and decoder input
            code_size (integer): dimensions of the latent space of the VAE
        """
        super(VAE, self).__init__()
        self.inner_shape = inner_shape
        self.code_size = code_size
        self.encoder = encoder
        self.encoder_fc = tf.keras.layers.Dense(code_size*2, name='encoder_fc')
        self.encoder_fc(tf.keras.Input(self.inner_shape, name='encoder_fc_input'))
        self.decoder_fc = tf.keras.layers.Dense(self.inner_shape, name='decoder_fc')
        self.decoder_fc(tf.keras.Input((self.code_size,), name='decoder_fc_input'))
        self.decoder = decoder
        
    def encode(self, x):
        x = self.encoder(x)
        x = tf.keras.layers.Flatten()(x)
        return tf.split(self.encoder_fc(x), num_or_size_splits=2, axis=1)
    
    def sample(self, mu, logvar, training=False):
        """
        Operation to get a sample from the normal distribution defined
        by mu and sigma. Sampling is only done during training.
        Args:
            mu (tf.Tensor): mean of the normal distribution
            logvar (tf.Tensor): log variance of the normal distribution
            training (bool, optional): if the model is training, defaults to False

        Returns:
            tf.Tensor: a sample from the normal distribution defined
            by mu and sigma if training, or simply mu if not
        """
        if training:
            return mu + tf.math.exp(0.5 * logvar) * tf.random.normal(shape=(self.code_size,))
        else:
            return mu
  
    def decode(self, z):
        z = self.decoder_fc(z)
        z = tf.keras.layers.Reshape((self.inner_shape,))(z)
        return self.decoder(z)
  
    def call(self, x, training=False):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar, training)
        return mu, logvar, self.decode(z)
    
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss
    
    def train_step(self, data):
        x_original, _ = data
        with tf.GradientTape() as tape:
            mu, logvar, x_recon = self(x_original, training=True)
            loss_value, recon_loss, kld_loss = self.loss(x_original, mu, logvar, x_recon)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"total_loss": loss_value, "recon_loss": recon_loss, "kld_loss":kld_loss}

def elbo_loss_generator(beta=0.1):
    """
    Generator for elbo loss function for VAE.
    Args:
        beta (float, optional): Weight of the KL divergence as shown
        in https://openreview.net/forum?id=Sy2fzU9gl, defaults to 0.1
    """
    def elbo_loss(x_original, mu, logvar, x_recon):
        """
        Evidence lower bound function, composed of a reconstruction term plus
        a KL divergence to regularize the generated latent space.
        Args:
            x_original (tf.Tensor): original images
            mu (tf.Tensor): mean of the normal distribution
            logvar (tf.Tensor): log variance of the normal distribution
            x_recon (tf.Tensor): reconstructed images

        Returns:
            tuple of tf.Tensors: tuple with total loss, reconstruction loss and KLD loss
        """
        image_height, image_width, image_channels = x_original.shape[1:]
        recon_loss = tf.keras.losses.binary_crossentropy(
            tf.reshape(x_original, (-1, image_height*image_width*image_channels)),
            tf.reshape(x_recon, (-1, image_height*image_width*image_channels))
        )
        recon_loss = tf.reduce_mean(recon_loss)
        kld_loss = -0.5 * (1+logvar-tf.square(mu)-tf.exp(logvar))
        kld_loss = tf.reduce_mean(kld_loss)
        return recon_loss+beta*kld_loss, recon_loss, kld_loss
    return elbo_loss
