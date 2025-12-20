import tensorflow as tf
import keras
import numpy as np

# Function to print 2D tensor
def printModelParam(cnn):
    for layer in cnn.layers:
        list_arr = layer.get_weights()
        i = 0
        for arr in list_arr:
            i += 1
            if len(arr) == 0:
                continue
            with open('weight/' + layer.name + '_' + str(i) + '.txt', 'w') as f:
                for ind, val in enumerate(arr.shape):
                    if ind == len(arr.shape) - 1:
                        print(val, file = f)
                    else:
                        print(val, file = f, end = ',')
                # Check the number of dimensions of the current weight array
                if arr.ndim == 4:
                    # For kernel (H, W, IN_CHANNEL, OUT_CHANNEL)
                    for out_channel in range(arr.shape[3]):
                        for in_channel in range(arr.shape[2]):
                            for h in range(arr.shape[0]):
                                for w in range(arr.shape[1]):
                                    if w == arr.shape[1] - 1:
                                        print(arr[h, w, in_channel, out_channel], file = f)
                                    else:
                                        print(arr[h, w, in_channel, out_channel], end=',', file = f)
                elif arr.ndim == 2:
                    for d0 in range(arr.shape[0]):
                        for d1 in range(arr.shape[1]):
                            if d1 == arr.shape[1] - 1:
                                print(arr[d0, d1], file = f)
                            else:
                                print(arr[d0, d1], end = ',', file = f)
                elif arr.ndim == 1:
                    for d0 in range(arr.shape[0]):
                        if d0 == arr.shape[0] - 1:
                            print(arr[d0], file = f)
                        else:
                            print(arr[d0], end = ',', file = f)
                else:
                    print(arr, file = f)
# Xay dung class CNN ke thua tu model
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        x = input
        for ind, layer in enumerate(self.layers):
            if ind == 0:
              with open('input.txt', 'w') as f:
                print('intput')
                for ind, dim in enumerate(x.shape):
                    if ind == len(x.shape) - 1:
                      print(dim, file = f)
                    else:
                      print(dim, file = f, end = ',')
                if x.ndim == 4:
                    for N in range(x.shape[0]):
                      for C in range(x.shape[3]):
                        for H in range(x.shape[1]):
                          for W in range(x.shape[2]):
                            if W == x.shape[2] - 1:
                              print(x[N, H, W, C], file = f)
                            else:
                              print(x[N, H, W, C], file = f, end = ',')
            else:
              with open('output/' + layer.name + '.output' + '.txt', 'w') as f:
                print(layer.name + 'output')
                x = layer(x)
                x_numpy = x.numpy()
                if x.ndim == 4:
                  for ind, dim in enumerate(x_numpy.shape):
                    if ind == len(x_numpy.shape) - 1:
                      print(dim, file = f)
                    else:
                      print(dim, file = f, end = ',')
                  for N in range(x_numpy.shape[0]):
                    for C in range(x_numpy.shape[3]):
                      for H in range(x_numpy.shape[1]):
                        for W in range(x_numpy.shape[2]):
                          if W == x_numpy.shape[2] - 1:
                            print(x_numpy[N, H, W, C], file = f)
                          else:
                            print(x_numpy[N, H, W, C], file = f, end = ',')
                elif x.ndim == 2:
                  for ind, dim in enumerate(x.shape):
                    if ind == len(x.shape) - 1:
                      print(dim, file = f)
                    else:
                      print(dim, file = f, end = ',')
                  for N in range(x_numpy.shape[0]):
                    for C in range(x_numpy.shape[1]):
                      if C == x.shape[1] - 1:
                        print(x_numpy[N, C], file = f)
                      else:
                        print(x_numpy[N, C], file = f, end = ',')
        return x
            
def build_CNN_model():
    inputs = keras.Input(shape=(32, 32, 3))  
    
    x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    

    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(10, activation='softmax')(x)  # Output layer for 10 classes
    model = CustomModel(inputs=inputs, outputs=outputs, name='cifar_10_cnn')
    return model

# Tạo đối tượng dataset sẵn có
if __name__ == '__main__':
    cifar_10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar_10.load_data()
    x_train, x_test = x_train/255, x_test/255
    cnn = build_CNN_model()
    cnn.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn.load_weights('cifar_10_cnn.weights.h5')
    # printModelParam(cnn)
    # cnn.forward(x_test[0].reshape(-1, 32, 32, 3))
    cnn.summary()