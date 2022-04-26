'''VGG11/13/16/19 in keras.'''
import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
import random
import visualkeras

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


trainable_layers = {
    'VGG11': [0, 2, 4, 5, 7, 8, 10, 11],
    'VGG13': [0, 1, 3, 4, 6, 7, 9, 10, 11, 12],
    'VGG16': [0, 1, 3, 4,  6, 7, 8,  10, 11, 12, 14, 15, 16],
    'VGG19': [0, 1,  3, 4,  6, 7, 8, 9,  11, 12, 13, 14,  16, 17, 18, 19],
}


def create_seed_model(input_shape=(32,32,3), dimension='VGG16', trainedLayers=0):

    num_classes = 10
    lay_count = 0

    if trainedLayers > 0:

        randomlist = random.sample(trainable_layers[dimension], trainedLayers)
        print(randomlist)

        with open('/app/layers.txt', '+a') as f:
            print(randomlist, file=f)

        model = Sequential()
        model.add(tensorflow.keras.Input(shape=input_shape))
        for x in cfg[dimension]:
            if x == 'M':
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                if lay_count in randomlist:
                    model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
                    model.add(BatchNormalization(trainable=True))
                    model.add(Activation(activations.relu))
                else:
                    model.add(Conv2D(x, (3, 3), padding='same', trainable=False))
                    model.add(BatchNormalization(trainable=False))
                    model.add(Activation(activations.relu))
            lay_count += 1

        #model.add(Flatten())
        model.add(AveragePooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

        print(" --------------------------------------- ")
        print(" ------------------MODEL CREATED------------------ ")
        print(" --------------------------------------- ")

    else:
        model = Sequential()
        model.add(tensorflow.keras.Input(shape=input_shape))
        for x in cfg[dimension]:
            if x == 'M':
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                print("trani: ", x)
                model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
                model.add(BatchNormalization(trainable=True))
                model.add(Activation(activations.relu))

        # model.add(Flatten())
        model.add(AveragePooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

    return model


### generate a full trainable seed model

# def create_seed_model(input_shape=(32, 32, 3), dimension='VGG16'):
#     num_classes = 10
#     model = Sequential()
#     model.add(keras.Input(shape=input_shape))
#     for x in cfg[dimension]:
#         if x == 'M':
#             model.add(MaxPooling2D(pool_size=(2, 2)))
#         else:
#             print("trani: ", x)
#             model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
#             model.add(BatchNormalization(trainable=True))
#             model.add(Activation(activations.relu))
#
#     # model.add(Flatten())
#     model.add(AveragePooling2D(pool_size=(1, 1)))
#     model.add(Flatten())
#     model.add(Dense(num_classes, activation='softmax'))
#     opt = keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt, metrics=['accuracy'])
#
#     return model


model = create_seed_model(dimension='VGG16',trainedLayers=0)  # ,trainL=layers['trainedLayers'])

visualkeras.layered_view(model, legend=True) # without custom font
from PIL import ImageFont
# font = ImageFont.truetype("arial.ttf", 12)
font = ImageFont.load_default()
visualkeras.layered_view(model, to_file='figures/vgg16_legend_jetson.png', type_ignore=[visualkeras.SpacingDummyLayer],legend=True, font=font)
visualkeras.layered_view(model,type_ignore=[visualkeras.SpacingDummyLayer],legend=True, font=font).show() # selected font


# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='figures/model_plot.png', show_shapes=True, show_layer_names=True)
# visualkeras.graph_view(model).show()





# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D
# from collections import defaultdict
# import visualkeras
# from PIL import ImageFont
#
# # create VGG16
# image_size = 224
# model = Sequential()
# model.add(InputLayer(input_shape=(image_size, image_size, 3)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
# model.add(visualkeras.SpacingDummyLayer())
#
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
# model.add(visualkeras.SpacingDummyLayer())
#
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
# model.add(visualkeras.SpacingDummyLayer())
#
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(visualkeras.SpacingDummyLayer())
#
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
# model.add(MaxPooling2D())
# model.add(visualkeras.SpacingDummyLayer())
#
# model.add(Flatten())
#
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1000, activation='softmax'))
#
# # Now visualize the model!
#
# color_map = defaultdict(dict)
# color_map[Conv2D]['fill'] = 'orange'
# color_map[ZeroPadding2D]['fill'] = 'gray'
# color_map[Dropout]['fill'] = 'pink'
# color_map[MaxPooling2D]['fill'] = 'red'
# color_map[Dense]['fill'] = 'green'
# color_map[Flatten]['fill'] = 'teal'
#
# # font = ImageFont.truetype("arial.ttf", 32)
# font = ImageFont.load_default()
# visualkeras.layered_view(model, to_file='figures/vgg16.png', type_ignore=[visualkeras.SpacingDummyLayer])
# visualkeras.layered_view(model, to_file='figures/vgg16_legend.png', type_ignore=[visualkeras.SpacingDummyLayer],
#                          legend=True, font=font)
# visualkeras.layered_view(model, to_file='figures/vgg16_spacing_layers.png', spacing=0)
# visualkeras.layered_view(model, to_file='figures/vgg16_type_ignore.png',
#                          type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.SpacingDummyLayer])
# visualkeras.layered_view(model, to_file='figures/vgg16_color_map.png',
#                          color_map=color_map, type_ignore=[visualkeras.SpacingDummyLayer])
# visualkeras.layered_view(model, to_file='figures/vgg16_flat.png',
#                          draw_volume=False, type_ignore=[visualkeras.SpacingDummyLayer])
# visualkeras.layered_view(model, to_file='figures/vgg16_scaling.png',
#                          scale_xy=1, scale_z=1, max_z=1000, type_ignore=[visualkeras.SpacingDummyLayer])