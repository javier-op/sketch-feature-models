import tensorflow as tf
import numpy as np

class BYOL(tf.keras.Model):
    """
    Bootstrap Your Own Latent  wrapper implementation for unsupervised
    feature extraction, where an online network tries to predict the
    output of a target network. Both networks receive different
    transformations of the same input, the online network is updated
    through gradients and the target network is updated as a moving
    average of the online network.
    https://arxiv.org/abs/2006.07733
    """

    def __init__(self, encoder, projector, predictor, tau, t_func_online, t_func_target):
        """
        BYOL constructor
        Args:
            encoder (tf.Model): encoder architecture
            projector (tf.Model): projector architecture
            predictor (tf.Model): predictor architecture
            tau (float): target decay rate for moving average,
                         values between 0 and 1, 0 means the target
                         network copies the values of the online
                         network, 1 means it stops being updated,
                         expects values close to 1 for slow updates.
            t_func_online (callable): transformation for online network
            t_func_target (callable): transformation for target network
        """
        super(BYOL, self).__init__()
        self.online_encoder = encoder
        self.online_encoder._name = 'online_encoder'
        self.target_encoder = tf.keras.models.clone_model(self.online_encoder)
        self.target_encoder._name = 'target_encoder'
        self.online_projector = projector
        self.online_projector._name = 'online_projector'
        self.target_projector = tf.keras.models.clone_model(self.online_projector)
        self.target_projector._name = 'target_projector'
        self.online_predictor = predictor
        self.online_predictor._name = 'online_predictor'
        self.tau = tau
        self.t_func_online = t_func_online
        self.t_func_target = t_func_target
    
    def call(self, x, training=False):
        if training:
            if self.t_func_online is not None:
                t_online = self.t_func_online(x)
            else:
                t_online = x
            if self.t_func_target is not None:
                t_target = self.t_func_target(x)
            else:
                t_target = x
            target_representation = self.target_encoder(t_target, training=True)
            target_projection = self.target_projector(target_representation, training=True)
            target_projection = tf.stop_gradient(target_projection)
            online_representation = self.online_encoder(t_online, training=True)
            online_projection = self.online_projector(online_representation, training=True)
            online_prediction = self.online_predictor(online_projection, training=True)
            return target_projection, online_prediction
        else:
            online_representation = self.online_encoder(x, training=False)
            return online_representation
    
    def update_moving_average(self):
        online_encoder_weights = np.array(self.online_encoder.get_weights())
        target_encoder_weights = np.array(self.target_encoder.get_weights())
        target_encoder_weights = self.tau*target_encoder_weights + (1-self.tau)*online_encoder_weights
        self.target_encoder.set_weights(target_encoder_weights)
        online_projector_weights = np.array(self.online_projector.get_weights())
        target_projector_weights = np.array(self.target_projector.get_weights())
        target_projector_weights = self.tau*target_projector_weights + (1-self.tau)*online_projector_weights
        self.target_projector.set_weights(target_projector_weights)
    
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss
    
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            target_projection, online_prediction = self(x, training=True)
            loss_value = self.loss(target_projection, online_prediction)
        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.update_moving_average()
        return {"loss": loss_value}


def byol_loss(target_projection, online_prediction):
    """
    The loss function for BYOL is describes as a simple cosine similarity
    as shown in https://arxiv.org/abs/2006.07733
    Args:
        target_projection (tf.Tensor): output from target network
        online_prediction (tf.Tensor): output from online network

    Returns:
        tf.Tensor: total loss
    """
    loss = tf.keras.losses.cosine_similarity(target_projection, online_prediction, axis=1)
    loss = tf.reduce_mean(loss)
    return loss
