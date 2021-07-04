import tensorflow as tf

class M2SSVAE(tf.keras.Model):
    """
    M2 model for a semi supervised VAE, the decoder receives both
    the output from the encoder and the output from the classifier,
    the idea is to separate the class information from the encoder
    so it extracts only "style" features.
    https://arxiv.org/abs/1406.5298
    """
    def __init__(self, encoder, decoder, inner_shape, code_size, classifier, classifier_output_shape, n_classes):
        """
        M2SSVAE constructor.
        Args:
            encoder (tf.Model): encoder architecture
            decoder (tf.Model): decoder architecture
            inner_shape (integer): size of encoder output
            code_size (integer): dimensions of the latent space of the M2SSVAE
            classifier (tf.Model): classifier architecture
            classifier_output_shape (integer): size of classifier output
            n_classes (integer): number of classes for final layer
        """
        super(M2SSVAE, self).__init__()
        self.inner_shape = inner_shape
        self.code_size = code_size
        self.classifier_output_shape = classifier_output_shape
        self.n_classes = n_classes
        self.encoder = encoder
        self.encoder_fc = tf.keras.layers.Dense(code_size*2, name='encoder_fc')
        self.encoder_fc(tf.keras.Input(self.inner_shape, name='encoder_fc_input'))
        self.decoder_fc = tf.keras.layers.Dense(self.inner_shape, name='decoder_fc')
        self.decoder_fc(tf.keras.Input(self.code_size+self.classifier_output_shape, name='decoder_fc_input'))
        self.decoder = decoder
        self.classifier = classifier
        self.classifier_fc = tf.keras.layers.Dense(n_classes, activation='softmax', name='classifier_fc')
        self.classifier_fc(tf.keras.Input(self.classifier_output_shape, name='classifier_fc_input'))
        
    def encode(self, x):
        x = self.encoder(x)
        x = tf.keras.layers.Flatten()(x)
        mu, logvar = tf.split(self.encoder_fc(x), num_or_size_splits=2, axis=1)
        return mu, logvar
    
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
            return mu + tf.exp(0.5 * logvar) * tf.random.normal(shape=(self.code_size,))
        else:
            return mu
    
    def decode(self, z, y):
        """
        Reconstructs the input using both infered style "z" and label "y"
        Args:
            z (tf.Tensor): encoder output, style "z"
            y (tf.Tensor): label "y"

        Returns:
            tf.Tensor: reconstructed input
        """
        z = self.decoder_fc(tf.concat([z, y], axis=1))
        z = tf.keras.layers.Reshape(self.inner_shape)(z)
        return self.decoder(z)
  
    def call(self, inputs, training=False):
        x, y = inputs
        x_features = self.classifier(x)
        y_pred = self.classifier_fc(x_features)
        if y is None:
            y = y_pred
        else:
            y = tf.one_hot(y, self.n_classes)
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar, training)
        return mu, logvar, self.decode(z, x_features), x_features, y_pred
    
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss
    
    def train_step(self, data):
        (x_original_l, y_true_l), (x_original_u, y_true_u) = data
        with tf.GradientTape() as tape:
            mu_l, logvar_l, x_recon_l, x_features_l, y_pred_l = self((x_original_l, y_true_l), training=True)
            mu_u, logvar_u, x_recon_u, x_features_u, y_pred_u = self((x_original_u, None), training=True)
            total_loss, recon_loss, kld_loss, categorical_loss = self.loss(
                x_original_l, y_true_l, mu_l, logvar_l, x_recon_l, y_pred_l,
                x_original_u, mu_u, logvar_u, x_recon_u, y_pred_u,
                self.n_classes
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"total_loss": total_loss, "recon_loss": recon_loss, "kld_loss": kld_loss, "categorical_loss": categorical_loss}


def labeled_loss(x_original, x_recon, mu, logvar, y_true, y_pred, n_classes, alpha, beta):
    """
    Labeled loss, regular ELBO loss plus a weighted classifier loss.
    Args:
        x_original (tf.Tensor): original images
        x_recon (tf.Tensor): reconstructed images
        mu (tf.Tensor): mean of the normal distribution
        logvar (tf.Tensor): log variance of the normal distribution
        y_true (tf.Tensor): real labels of the images
        y_pred (tf.Tensor): predicted labels
        n_classes (tf.Tensor): number of classes in the model
        alpha (float): Weight of the classifier loss as shown
        in https://arxiv.org/abs/1406.5298, defaults to 1.
        beta (float): Weight of the KL divergence as shown
        in https://openreview.net/forum?id=Sy2fzU9gl, defaults to 1.

    Returns:
        tuple of tf.Tensors: tuple with total loss, reconstruction loss, KLD loss and classifier loss
    """
    image_height, image_width, image_channels = x_original.shape[1:]
    loglik_recon = tf.keras.losses.binary_crossentropy(
        tf.reshape(x_original, (-1, image_height*image_width*image_channels)),
        tf.reshape(x_recon, (-1, image_height*image_width*image_channels))
    )
    kld = -0.5 * (1+logvar-tf.square(mu)-tf.exp(logvar))
    kld = tf.reduce_mean(kld, axis=1)
    _L = loglik_recon + beta*kld
    y_true = tf.one_hot(y_true, n_classes)
    categorical = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return _L + alpha*categorical, loglik_recon, kld, categorical


def unlabeled_loss(x_original, x_recon, mu, logvar, y_pred, beta):
    """
    Unlabeled loss, ponderated ELBO loss plus a entropy of predicted labels.
    Args:
        x_original (tf.Tensor): original images
        x_recon (tf.Tensor): reconstructed images
        mu (tf.Tensor): mean of the normal distribution
        logvar (tf.Tensor): log variance of the normal distribution
        y_pred (tf.Tensor): predicted labels
        beta (float): Weight of the KL divergence as shown
        in https://openreview.net/forum?id=Sy2fzU9gl, defaults to 1.

    Returns:
        tuple of tf.Tensors: tuple with total loss, reconstruction loss and KLD loss
    """
    image_height, image_width, image_channels = x_original.shape[1:]
    loglik_recon = tf.keras.losses.binary_crossentropy(
        tf.reshape(x_original, (-1, image_height*image_width*image_channels)),
        tf.reshape(x_recon, (-1, image_height*image_width*image_channels))
    )
    kld = -0.5 * (1+logvar-tf.square(mu)-tf.exp(logvar))
    kld = tf.reduce_mean(kld, axis=1)
    _L = loglik_recon + beta*kld
    entropy = tf.keras.losses.categorical_crossentropy(y_pred, y_pred)
    sum_y_pred_L = tf.reduce_sum(y_pred*tf.expand_dims(_L,1), axis=1)
    return sum_y_pred_L + entropy, loglik_recon, kld


def semi_supervised_loss_generator(alpha, beta):
    """
    Generator for loss function for NaiveSSVAE.
    Args:
        alpha (float): Weight of the classifier loss as shown
        in https://arxiv.org/abs/1406.5298, defaults to 1.
        beta (float): Weight of the KL divergence as shown
        in https://openreview.net/forum?id=Sy2fzU9gl, defaults to 1.
    """
    def semi_supervised_loss(x_original_l, y_true_l, mu_l, logvar_l, x_recon_l, y_pred_l,
                             x_original_u, mu_u, logvar_u, x_recon_u, y_pred_u,
                             n_classes):
        """
        Loss function for M2SSVAE, receives labeled and unlabeled batches separately.
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
            y_pred_u (tf.Tensor): predicted labels, unlabeled case
            n_classes (integer): number of classes in the model

        Returns:
            tuple of tf.Tensors: tuple with total loss, reconstruction loss, KLD loss and classification loss
        """
        loss_l, loglik_recon_l, kld_l, categorical = labeled_loss(x_original_l, x_recon_l, mu_l, logvar_l, y_true_l, y_pred_l, n_classes, alpha, beta)
        loss_u, loglik_recon_u, kld_u = unlabeled_loss(x_original_u, x_recon_u, mu_u, logvar_u, y_pred_u, beta)
        total_loss = tf.reduce_mean(tf.concat([loss_l, loss_u], axis=0), axis=0)
        total_recon_loss = tf.reduce_mean(tf.concat([loglik_recon_l, loglik_recon_u], axis=0), axis=0)
        total_kld_loss = tf.reduce_mean(tf.concat([kld_l, kld_u], axis=0), axis=0)
        categorical_loss = tf.reduce_mean(categorical, axis=0)
        return total_loss, total_recon_loss, total_kld_loss, categorical_loss
    return semi_supervised_loss
