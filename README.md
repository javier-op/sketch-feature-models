# sketch-feature-models
Tensorflow implementations of some deep learning models for feature extraction in sketches.

All models, beside AlexNet based ones, require sub architectures to be built, compiled and then passed in the constructor. VAE based architectures usually have an encoder and decoder, in which the decoder is a reversed encoder, for that you can find an AlexNetEncoder and AlexNetDecoder in the alexnet_model module, you can also build your own architectures.

Usage example:
```python
from models.alexnet_model import AlexNetEncoder, AlexNetDecoder
from models.vae_model import VAE, elbo_loss_generator
import tensorflow as tf

vae_input = tf.keras.Input((256,256,1), name='vae_input')
encoder = AlexNetEncoder()
encoder(vae_input)

decoder_input = tf.keras.Input((1024,), name='decoder_input')
decoder = AlexNetDecoder()
decoder(decoder_input)

model = VAE(
    encoder,
    decoder,
    inner_shape=1024,
    code_size=32)
model(vae_input)

elbo_loss = elbo_loss_generator(beta=0.1)
model.compile(
    optimizer='adam',
    loss=elbo_loss,
)

# build a dataset that outputs (x_original, y_true)
train_dataset = buildTrainDataset()

model.fit(train_dataset, epochs=30)
```

Both semi supervised models, NaiveSSVAE and M2SSVAE, expect joint dataset with a batch of ((x_original_l, y_true_l), (x_original_u, y_true_u)). You can easily build that like this:

```python
labeled_train = buildLabeledTrainDataset()
unlabeled_train = buildUnlabebeledTrainDataset()
joint_dataset = tf.data.Dataset.zip((labeled_train, unlabeled_train))
```