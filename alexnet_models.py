import tensorflow as tf

class AlexNet(tf.keras.Model):
    """
    AlexNet implementation with subtle variations.
    """
    def __init__(self, number_of_classes):
        """
        AlexNet constructor.
        Args:
            number_of_classes (integer): size of the final one hot encoded layer
        """
        super(AlexNet, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(96, (11,11), strides = 4, padding = 'valid',  kernel_initializer = 'he_normal')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'valid')
        self.relu = tf.keras.layers.ReLU()  
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(256, (5,5), padding = 'valid',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(384, (3,3), padding = 'valid', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()
        
        self.conv_4 = tf.keras.layers.Conv2D(384, (3,3), padding = 'valid', kernel_initializer='he_normal')
        self.bn_conv_4 = tf.keras.layers.BatchNormalization()
        
        self.conv_5 = tf.keras.layers.Conv2D(256, (3,3), padding = 'valid', kernel_initializer='he_normal')
        self.bn_conv_5 = tf.keras.layers.BatchNormalization()
        
        self.fc6 = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')        
        self.bn_fc_6 = tf.keras.layers.BatchNormalization()
        
        self.fc7 = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')
        self.bn_fc_7 = tf.keras.layers.BatchNormalization()
        
        self.fc8 = tf.keras.layers.Dense(number_of_classes) if number_of_classes else None

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_conv_1(x)
        x = self.relu(x) 
        x = self.max_pool(x)

        x = self.conv_2(x)
        x = self.bn_conv_2(x) 
        x = self.relu(x) 
        x = self.max_pool(x)

        x = self.conv_3(x)
        x = self.bn_conv_3(x)
        x = self.relu(x)  

        x = self.conv_4(x)
        x = self.bn_conv_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.bn_conv_5(x)
        x = self.relu(x)
        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc6(x) 
        x = self.bn_fc_6(x) 
        x = self.relu(x) 
        
        x = self.fc7(x) 
        x = self.bn_fc_7(x) 
        x = self.relu(x)        

        if self.fc8:
            x = self.fc8(x)
        return x


class AlexNetEncoder(tf.keras.Model):
    """
    AlexNet based encoder architecture with different padding so it can be reversed,
    6 convolutional layers followed by 2 fully connected layers.
    """
    def __init__(self):
        super(AlexNetEncoder, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(96, (11,11), strides = 4, padding = 'same',  kernel_initializer = 'he_normal')
        self.max_pool = tf.keras.layers.MaxPool2D((2,2), padding = 'same')
        self.relu = tf.keras.layers.ReLU()
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(256, (5,5), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()
        
        self.conv_3 = tf.keras.layers.Conv2D(256, (5,5), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()

        self.conv_4 = tf.keras.layers.Conv2D(384, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_4 = tf.keras.layers.BatchNormalization()
        
        self.conv_5 = tf.keras.layers.Conv2D(384, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_5 = tf.keras.layers.BatchNormalization()
        
        self.conv_6 = tf.keras.layers.Conv2D(256, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_6 = tf.keras.layers.BatchNormalization()
        
        self.fc7 = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')        
        self.bn_fc_7 = tf.keras.layers.BatchNormalization()
        
        self.fc8 = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')
        self.bn_fc_8 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)  
        x = self.bn_conv_1(x, training)
        x = self.relu(x) 
        x = self.max_pool(x) 

        x = self.conv_2(x)
        x = self.bn_conv_2(x, training) 
        x = self.relu(x) 
        x = self.max_pool(x)
        
        x = self.conv_3(x)
        x = self.bn_conv_3(x, training) 
        x = self.relu(x) 
        x = self.max_pool(x)
        
        x = self.conv_4(x)
        x = self.bn_conv_4(x, training)
        x = self.relu(x)  

        x = self.conv_5(x)
        x = self.bn_conv_5(x, training)
        x = self.relu(x)

        x = self.conv_6(x)
        x = self.bn_conv_6(x, training)
        x = self.relu(x)
         
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc7(x) 
        x = self.bn_fc_7(x, training) 
        x = self.relu(x) 
        
        x = self.fc8(x) 
        x = self.bn_fc_8(x, training) 
        x = self.relu(x)        

        return x


class AlexNetDecoder(tf.keras.Model):
    """
    AlexNet based decoder architecture, an inverted AlexNetEncoder,
    2 fully connected layers followed by 6 transposed convolutional layers.
    """
    def __init__(self):
        super(AlexNetDecoder, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.sigmoid = tf.keras.activations.sigmoid
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        
        self.fc1 = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')
        self.bn_fc_1 = tf.keras.layers.BatchNormalization()
        
        self.fc2 = tf.keras.layers.Dense(8*8*256, kernel_initializer='he_normal')        
        self.bn_fc_2 = tf.keras.layers.BatchNormalization()
        
        self.conv_3 = tf.keras.layers.Conv2DTranspose(384, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()
        
        self.conv_4 = tf.keras.layers.Conv2DTranspose(384, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_4 = tf.keras.layers.BatchNormalization()
        
        self.conv_5 = tf.keras.layers.Conv2DTranspose(256, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_5 = tf.keras.layers.BatchNormalization()
        
        self.conv_6 = tf.keras.layers.Conv2DTranspose(256, (5,5), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_6 = tf.keras.layers.BatchNormalization()
        
        self.conv_7 = tf.keras.layers.Conv2DTranspose(96, (5,5), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_7 = tf.keras.layers.BatchNormalization()
        
        self.conv_8 = tf.keras.layers.Conv2DTranspose(1, (11,11), strides = 4, padding = 'same',  kernel_initializer = 'he_normal') 
        self.bn_conv_8 = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.fc1(inputs) 
        x = self.bn_fc_1(x, training) 
        x = self.relu(x)  
        
        x = self.fc2(x) 
        x = self.bn_fc_2(x, training) 
        x = self.relu(x)
        x = tf.keras.layers.Reshape((8,8,256))(x)
        
        x = self.conv_3(x)
        x = self.bn_conv_3(x, training)
        x = self.relu(x)
        
        x = self.conv_4(x)
        x = self.bn_conv_4(x, training)
        x = self.relu(x)
        
        x = self.conv_5(x)
        x = self.bn_conv_5(x, training)
        x = self.relu(x)
        
        x = self.upsample(x)
        x = self.conv_6(x)
        x = self.bn_conv_6(x, training)
        x = self.relu(x)
        
        x = self.upsample(x)
        x = self.conv_7(x)
        x = self.bn_conv_7(x, training)
        x = self.relu(x)
        
        x = self.upsample(x)
        x = self.conv_8(x)
        x = self.bn_conv_8(x, training)
        x = self.sigmoid(x)
        
        return x
