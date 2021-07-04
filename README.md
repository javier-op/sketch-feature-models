# sketch-feature-models
Tensorflow implementations of some deep learning models for feature extraction in sketches.

All models, beside AlexNet based ones, require sub architectures to be built, compiled and then passed in the constructor. VAE based architectures usually have an encoder and decoder, in which the decoder is a reversed encoder, for that you can find an AlexNetEncoder and AlexNetDecoder in the alexnet_model module, you can also build your own architectures.

Usage example:
```python
from models.alexnet_model import AlexNetEncoder, AlexNetDecoder
from models.vae_model import VAE
import tensorflow as tf

vae_input = tf.keras.Input((256,256,1), name='vae')
encoder = AlexNetEncoder()
encoder(vae_input)

decoder_input = tf.keras.Input((1024,), name='decoder')
decoder = AlexNetDecoder()
decoder(decoder_input)

model = VAE(
    encoder,
    decoder,
    inner_shape=1024,
    code_size=32)
model(vae_input)
```
