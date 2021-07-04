import tensorflow as tf

class NaiveSSVAE(tf.keras.Model):
    """
    Simple semi supervised VAE implementation, Y shaped network
    in which the output from the encoder is fed to the decoder in
    labeled and unlabeled cases, and a classifier only for labeled
    cases. Encoder, decoder and classifier architectures should be
    instantiated before and passed in the constructor.
    """
    def __init__(self, encoder, decoder, inner_shape, code_size, classifier, classifier_output_shape, n_classes):
        """
        NaiveSSVAE constructor.
        Args:
            encoder (tf.Model): encoder architecture
            decoder (tf.Model): decoder architecture
            inner_shape (integer): size of encoder output
            code_size (integer): dimensions of the latent space of the NaiveSSVAE
            classifier (tf.Model): classifier architecture, if None only
                                 a fully connected layer will be used
            classifier_output_shape (integer): size of encoder output
            n_classes (integer): number of classes for final layer
        """
        super(NaiveSSVAE, self).__init__()
        self.inner_shape = inner_shape
        self.code_size = code_size
        self.classifier_output_shape = classifier_output_shape
        self.n_classes = n_classes
        self.encoder = encoder
        self.encoder_fc = tf.keras.layers.Dense(code_size*2, name='encoder_fc')
        self.encoder_fc(tf.keras.Input(self.inner_shape, name='encoder_fc_input'))
        self.decoder_fc = tf.keras.layers.Dense(self.inner_shape, name='decoder_fc')
        self.decoder_fc(tf.keras.Input((self.code_size,), name='decoder_fc_input'))
        self.decoder = decoder
        self.classifier = classifier
        self.classifier_fc = tf.keras.layers.Dense(n_classes, activation='softmax', name='classifier_fc')
        if classifier is None:
            self.classifier_fc(tf.keras.Input((self.code_size,), name='classifier_fc_input'))
        else:
            self.classifier_fc(tf.keras.Input((self.classifier_output_shape,), name='classifier_fc_input'))
        
    def encode(self, x, training=False):
        x = self.encoder(x, training)
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
            return mu + tf.exp(0.5*logvar)*tf.random.normal(shape=(self.code_size,))
        else:
            return mu
  
    def decode(self, z, training=False):
        z = self.decoder_fc(z)
        z = tf.keras.layers.Reshape((self.inner_shape,))(z)
        return self.decoder(z, training)
  
    def call(self, x, training=False):
        mu, logvar = self.encode(x, training)
        z = self.sample(mu, logvar, training)
        x_recon = self.decode(z, training)
        if self.classifier is None:
            x_features = z
            y_pred = self.classifier_fc(x_features)
        else:
            x_features = self.classifier(x, training)
            y_pred = self.classifier_fc(x_features)
        return mu, logvar, x_recon, x_features, y_pred
    
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss
    
    def train_step(self, data):
        (x_original_l, y_true_l), (x_original_u, y_true_u) = data
        with tf.GradientTape() as tape:
            mu_l, logvar_l, x_recon_l, x_features_l, y_pred_l = self(x_original_l, training=True)
            mu_u, logvar_u, x_recon_u, x_features_u, y_pred_u = self(x_original_u, training=True)
            total_loss, recon_loss, kld_loss, categorical_loss = self.loss(x_original_l, y_true_l, mu_l, logvar_l, x_recon_l, y_pred_l,
                                                                           x_original_u, mu_u, logvar_u, x_recon_u,
                                                                           self.n_classes)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"total_loss": total_loss, "recon_loss": recon_loss, "kld_loss": kld_loss, "categorical_loss": categorical_loss}


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
        tuple of tf.Tensors: tuple with reconstruction loss and KLD loss
    """
    recon_loss = tf.keras.losses.binary_crossentropy(tf.reshape(x_original, [-1]), tf.reshape(x_recon, [-1]))
    recon_loss = tf.reduce_mean(recon_loss)
    kld_loss = -0.5 * (1+logvar-tf.square(mu)-tf.exp(logvar))
    kld_loss = tf.reduce_mean(kld_loss)
    return recon_loss, kld_loss


def categorical_crossentropy(y_true, y_pred, n_classes):
    """
    Wrapper for categorical crossentropy that also transforms label
    to one hot encoded.
    Args:
        y_true (tf.Tensor): true label, not one hot encoded
        y_pred (tf.Tensor): predicted label
        n_classes (integer): number of classes in the model

    Returns:
        tf.Tensor: categorical crossentropy loss
    """
    y_true = tf.one_hot(y_true, n_classes)
    categorical_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    categorical_loss = tf.reduce_mean(categorical_loss)
    return categorical_loss


def ss_loss_generator(beta=1, gamma=1):
    """
    Generator for elbo loss function for NaiveSSVAE.
    Args:
        beta (integer, optional): Weight of the KL divergence as shown
        in https://openreview.net/forum?id=Sy2fzU9gl, defaults to 1.
        gamma (integer, optional): Weight of the categorical loss in the
        labeled case, defaults to 1.
    """
    def semi_supervised_loss(x_original_l, y_true_l, mu_l, logvar_l, x_recon_l, y_pred_l,
                             x_original_u, mu_u, logvar_u, x_recon_u,
                             n_classes):
        """
        Loss function for SSVAE, receives labeled and unlabeled batches separately.
        Args:
            x_original_l (tf.Tensor): original images, labeled case
            y_true_l (tf.Tensor): real labels of the images, labeled case
            mu_l (tf.Tensor): mean of the normal distribution, labeled case
            logvar_l (tf.Tensor): log variance of the normal distribution, labeled case
            x_recon_l (tf.Tensor): reconstructed images, labeled case
            y_pred_l (tf.Tensor): predicted labels, labeled case
            x_original_u (tf.Tensor): original images, unlabeled case
            mu_u (tf.Tensor): mean of the normal distribution, unlabeled case
            logvar_u (tf.Tensor): log variance of the normal distribution, unlabeled case
            x_recon_u (tf.Tensor): reconstructed images, unlabeled case
            n_classes (tf.Tensor): number of classes in the model

        Returns:
            tuple of tf.Tensors: tuple with total loss, reconstruction loss, KLD loss and classification loss
        """
        x_original = tf.cond(tf.size(x_original_l)>0, lambda: tf.concat([x_original_l, x_original_u], axis=0), lambda: x_original_u)
        mu = tf.cond(tf.size(mu_l)>0, lambda: tf.concat([mu_l, mu_u], axis=0), lambda: mu_u)
        logvar = tf.cond(tf.size(logvar_l)>0, lambda: tf.concat([logvar_l, logvar_u], axis=0), lambda: logvar_u)
        x_recon = tf.cond(tf.size(x_recon_l)>0, lambda: tf.concat([x_recon_l, x_recon_u], axis=0), lambda: x_recon_u)
        recon_loss, kld_loss = elbo_loss(x_original, mu, logvar, x_recon)
        categorical_loss = tf.cond(tf.size(y_true_l)>0, lambda: categorical_crossentropy(y_true_l, y_pred_l, n_classes), lambda: 0.)
        return recon_loss+beta*kld_loss+gamma*categorical_loss, recon_loss, kld_loss, categorical_loss
    return semi_supervised_loss
