from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.initializers import lecun_normal
from keras.optimizers import RMSprop
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from time import time
import random
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from keras_tqdm import TQDMCallback
#
# Load Dataset
#


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def fast_load_dataset(path):
    dog_files = np.load(path + '/files.npy')
    dog_targets = np.load(path + '/targets.npy')
    return dog_files, dog_targets


def load_and_save(path):
    dog_files, dog_targets = load_dataset(path)
    print('saving files...')
    np.save(path + '/files.npy', dog_files)
    np.save(path + '/targets.npy', dog_targets)
    print('files saved!')


def test_load_save():
    for path in ['dogImages/train', 'dogImages/valid', 'dogImages/test']:
        load_and_save(path)


#
# Supplied methods and helpers to load tensors
#
ImageFile.LOAD_TRUNCATED_IMAGES = True


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def load_tensors_supplied(paths):
    tensors = paths_to_tensor(paths).astype('float32') / 255
    return tensors


#
# Own methods and helpers to load tensors
#

def load_tensors(file_path):
    # NOTE: file_path should be one of ['dogImages/train', 'dogImages/valid', 'dogImages/test']
    tensors = np.load(file_path + '/tensors.npy')
    return tensors


def test_save_tensors():
    print("saving tensors...")
    paths = ['dogImages/train', 'dogImages/valid', 'dogImages/test']
    for path in paths:
        files, targets = fast_load_dataset(path)
        tensors = load_tensors_supplied(files)
        tensors_path = path + '/tensors.npy'
        np.save(tensors_path, tensors)
    print("saved all tensors")

#
# Own methods and helpers to load tensors (npz)
#


def load_tensors_targets(file_path):
    # NOTE: file_path should be one of ['dogImages/train', 'dogImages/valid', 'dogImages/test']
    tensors = np.load(file_path + '/tensors_targets.npz')['tensors']
    targets = np.load(file_path + '/tensors_targets.npz')['targets']
    return tensors, targets


def test_load_tensors_targets():
    paths = ['dogImages/train', 'dogImages/valid', 'dogImages/test']
    for path in paths:
        tensors_targets = np.load(path + '/tensors_targets.npz')
        print(tensors_targets.files)


def test_save_tensors_targets():
    print("saving tensors and targets...")
    paths = ['dogImages/train', 'dogImages/valid', 'dogImages/test']
    for path in ['dogImages/valid', 'dogImages/test']:
        files, targets = fast_load_dataset(path)
        tensors = load_tensors_supplied(files)
        np.savez(path + '/tensors_targets.npz', tensors=tensors, targets=targets)
        print(path, "successfully saved!")
    print("All tensors and targets saved!")


def dumb_save_tensors():
    print("saving tensors...")
    for path in ['dogImages/train', 'dogImages/valid', 'dogImages/test']:
        files, targets = fast_load_dataset(path)
        tensors = paths_to_tensor(files)
        np.savez(path+'/tensors.npz', tensors=tensors)
    print("All tensors saved!")


def dumb_load_tensors():
    print("loading tensors...")
    for path in ['dogImages/train', 'dogImages/valid', 'dogImages/test']:
        tensors = np.load(path + '/tensors.npz')['tensors']
        print(tensors.shape)
    print("All tensors loaded!")

#
# preprocess methods
#


def test_dumb_preprocess():
    # initalize Contrast-limited adaptive histogram equalization
    # clip limit governs ...
    # tileGridSize needs to be larger than first layer(s) kernel size
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    valid_files, valid_targets = fast_load_dataset('dogImages/valid')
    x = preprocess_img_to_tensor(valid_files[20], clahe)
    print(x.shape)


edge_flag = False


def preprocess_img_to_tensor(img_path, clahe):
    # loads RGB image as cv2 image
    rgb_img = cv2.imread(img_path)
    # converts RGB image to CIE L*a*b* color space
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    # split the CIE L*a*b* image into channels
    luminance, a_star, b_star = cv2.split(lab_img)
    # apply CLAHE to Luminance channel
    equalized_luminance = clahe.apply(luminance)
    # merge the channels back together
    merge = cv2.merge((equalized_luminance, a_star, b_star))
    # convert the CIE L*a*b* color space back into RGB
    equalized_img = cv2.cvtColor(merge, cv2.COLOR_Lab2BGR)
    # resize the image to (224, 224)
    resized_img = cv2.resize(equalized_img, (224, 224))

    if edge_flag:
        # extract the edges
        edges = cv2.Canny(resized_img, 100, 200)
        # convert rgb edge image to grayscale
        # gray_edges = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
        # merge edge channel into main image
        dumb_img = cv2.merge((resized_img, edges))
        # convert the image to 3D tensor with shape (224, 224, 4)
        x = image.img_to_array(dumb_img)
    else:
        # convert the image to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(resized_img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def preprocess_path(img_paths):
    # initalize Contrast-limited adaptive histogram equalization
    # clip limit governs ...
    # tileGridSize needs to be larger than first layer(s) kernel size
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    list_of_tensors = [preprocess_img_to_tensor(img_path, clahe) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def preprocess_tensors(paths):
    tensors = preprocess_path(paths).astype('float32') / 255
    return tensors

#
# keras model definitions
#


def hint_model_1(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    model = Sequential()

    # first layer
    model.add(Conv2D(filters=first_filters, input_shape=input_shape,
                     kernel_size=2, strides=1, activation='relu',
                     kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # second layer
    model.add(Conv2D(filters=first_filters * 2,
                     kernel_size=2, strides=1, activation='relu',
                     kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # third layer
    model.add(Conv2D(filters=first_filters * 4,
                     kernel_size=2, strides=1, activation='relu',
                     kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    #
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=number_of_labels, activation='softmax',
                    kernel_initializer=lecun_normal()))

    # set the filepath for saving the model's weights
    save_filepath = "saved_models/weights.best.hint_model_1.hdf5"
    return model, save_filepath


def hint_model_2(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    ks = 2  # kernel + stride example ks=2 means (2,2) kernel and stride of 2
    model = Sequential()

    model.add(Conv2D(filters=first_filters, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal(),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*2, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*4, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=number_of_labels, activation='softmax', kernel_initializer=lecun_normal()))

    filepath = "saved_models/weights.best.hint_model_2.hdf5"
    return model, filepath


def hint_model_2_bn(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    ks = 2  # kernel + stride example ks=2 means (2,2) kernel and stride of 2
    model = Sequential()

    model.add(Conv2D(filters=first_filters, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal(),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*2, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*4, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=number_of_labels, activation='softmax', kernel_initializer=lecun_normal()))

    filepath = "saved_models/weights.best.hint_model_2_bn.hdf5"
    return model, filepath


def hint_model_2_nin(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    ks = 2  # kernel + stride example ks=2 means (2,2) kernel and stride of 2
    model = Sequential()

    model.add(Conv2D(filters=first_filters, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal(), input_shape=input_shape))
    model.add(Conv2D(filters=first_filters, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*2, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal()))
    model.add(Conv2D(filters=first_filters*2, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*4, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal()))
    model.add(Conv2D(filters=first_filters*4, kernel_size=ks, strides=ks,
                     activation='relu', kernel_initializer=lecun_normal()))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=number_of_labels, activation='softmax', kernel_initializer=lecun_normal()))

    filepath = "saved_models/weights.best.hint_model_2_nin.hdf5"
    return model, filepath


def hint_model_2_nin_bn(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    ks = 2  # kernel + stride example ks=2 means (2,2) kernel and stride of 2
    model = Sequential()

    model.add(Conv2D(filters=first_filters, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal(), input_shape=input_shape))
    model.add(Conv2D(filters=first_filters, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*2, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal()))
    model.add(Conv2D(filters=first_filters*2, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=first_filters*4, kernel_size=1, strides=1,
                     kernel_initializer=lecun_normal()))
    model.add(Conv2D(filters=first_filters*4, kernel_size=ks, strides=ks,
                     kernel_initializer=lecun_normal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=number_of_labels, activation='softmax', kernel_initializer=lecun_normal()))

    filepath = "saved_models/weights.best.hint_model_2_nin_bn.hdf5"
    return model, filepath


def model_from_scratch(number_of_labels, first_filters=16, input_shape=(224, 224, 3)):
    model = Sequential()

    # first layer
    model.add(Conv2D(filters=first_filters, input_shape=input_shape, kernel_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    # second layer
    model.add(Conv2D(filters=first_filters*2, kernel_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    # third layer
    model.add(Conv2D(filters=first_filters*4, kernel_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=number_of_labels))
    # model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # set the filepath for saving the model's weights
    save_filepath = "saved_models/weights.best.from_scratch.hdf5"
    return model, save_filepath


#
#
#

def test_savez_targets():
    print("saving targets...")
    # load targets
    train = np.load('dogImages/train/targets.npy')
    test = np.load('dogImages/test/targets.npy')
    valid = np.load('dogImages/valid/targets.npy')
    # save targets
    np.savez('dogImages/targets.npz', train=train, test=test, valid=valid)
    print("all targets successfully saved!")


def test_savez_tensors():
    # load files
    train_files = np.load('dogImages/train/files.npy')
    test_files = np.load('dogImages/test/files.npy')
    valid_files = np.load('dogImages/valid/files.npy')
    # load tensors
    train = paths_to_tensor(train_files).astype('float32') / 255
    test = paths_to_tensor(test_files).astype('float32') / 255
    valid = paths_to_tensor(valid_files).astype('float32') / 255
    # save tensors
    print("saving tensors...")
    np.savez('dogImages/tensors.npz', train=train, test=test, valid=valid)
    print("all tensors successfully saved!")


def test_savez_tt():
    test_savez_targets()
    test_savez_tensors()


def test_load_tt():
    targets = np.load('dogImages/targets.npz')
    tensors = np.load('dogImages/tensors.npz')

    print(targets.files)
    print(tensors.files)

#
# helper functions
#


def load_data():
    # pre-process the data for Keras
    print('loading tensor and target data for keras...')
    start = time()
    raw_targets = np.load('dogImages/targets.npz')
    raw_tensors = np.load('dogImages/tensors.npz')
    stop = time()
    loading_time = round(stop - start, 0)
    print('targets and tensors loaded in %d seconds.\n' % loading_time)

    print('unpacking tensor and target data...')
    start = time()
    tensors = {'train': raw_tensors['train'],
               'test': raw_tensors['test'],
               'valid': raw_tensors['valid']}
    targets = {'train': raw_targets['train'],
               'test': raw_targets['test'],
               'valid': raw_targets['valid']}
    stop = time()
    unpacking_time = round(stop - start, 0)
    print('targets and tensors unpacked in %d seconds.\n' % unpacking_time)
    return tensors, targets


def train_model(model, save_path, tensors, targets, epochs=5, batch_size=20, resume=False):
    print("summarizing model...")
    model.summary()

    if resume:
        print("loading saved weights...")
        model.load_weights(save_path)

    start = time()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    stop = time()
    compilation_time = round(stop - start, 0)
    print('model compiled in %d seconds.\n' % compilation_time)

    print("unpacking tensors...")
    train_tensors, train_targets = tensors['train'], targets['train']
    valid_tensors, valid_targets = tensors['valid'], targets['valid']
    print("tensors unpacked!")

    checkpointer = ModelCheckpoint(filepath=save_path,
                                   verbose=1, save_best_only=True)
    start = time()
    model.fit(train_tensors, train_targets,
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=batch_size,
              callbacks=[checkpointer, TQDMCallback()], verbose=0)
    stop = time()
    fit_time = round(stop - start, 0)
    print('model fitted in 5 epochs over %d seconds.\n' % fit_time)


def test_model(model, save_path, test_tensors, test_targets):
    model.load_weights(save_path)

    # get index of predicted dog breed for each image in test set
    print('start preditions...')
    start = time()
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    stop = time()
    predict_time = round(stop - start, 0)
    print('predictions made in %d seconds.\n' % predict_time)

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(test_targets, axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%\n' % test_accuracy)

#
# main test loop
#


def summarize_model(model_func):
    model, save = model_func(133)
    model.summary()
    print(save)


def main_test(model_func=hint_model_2, epochs=5, batch_size=20, resume=False):
    # set random seed
    random.seed(8675309)

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    # define model and save path
    model, save_path = model_func(len(dog_names))

    tensors, targets = load_data()

    train_model(model, save_path, tensors, targets, epochs, batch_size, resume)

    test_model(model, save_path, tensors['test'], targets['test'])


def test_VGG16_bottleneck_features(epochs=5, batch_size=20):
    print('testing VGG16 bottleneck features')
    print('loading bottleneck features...')
    start = time()
    bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
    train_files, train_targets = fast_load_dataset('dogImages/train')
    valid_files, valid_targets = fast_load_dataset('dogImages/valid')
    test_files, test_targets = fast_load_dataset('dogImages/test')
    stop = time()
    loading_time = round(stop - start, 0)
    print('bottleneck features loaded in %d seconds.\n' % loading_time)

    print('extracting bottleneck tensors...')
    start = time()
    train_VGG16 = bottleneck_features['train']
    valid_VGG16 = bottleneck_features['valid']
    test_VGG16 = bottleneck_features['test']
    stop = time()
    extraction_time = round(stop - start, 0)
    print('bottleneck tensors extracted in %d seconds.\n' % extraction_time)

    print('defining model...')
    VGG16_model = Sequential()
    # VGG16_model.add(Dense(128, activation='relu', input_shape=train_VGG16.shape[1:], kernel_initializer=lecun_normal()))
    # VGG16_model.add(Dense(128, activation='relu', kernel_initializer=lecun_normal()))
    # VGG16_model.add(GlobalAveragePooling2D())
    VGG16_model.add(Flatten(input_shape=train_VGG16.shape[1:]))
    VGG16_model.add(Dense(133, activation='softmax', kernel_initializer=lecun_normal()))

    VGG16_model.summary()

    print('compiling model...')
    start = time()
    VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    stop = time()
    compiling_time = round(stop - start, 0)
    print('model compiled in %d seconds.\n' % compiling_time)

    print('training model...')
    start = time()
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                                   verbose=1, save_best_only=True)

    VGG16_model.fit(train_VGG16, train_targets,
                    validation_data=(valid_VGG16, valid_targets),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer, TQDMCallback()], verbose=0)
    stop = time()
    training_time = round(stop - start, 0)
    print('model trained in %d seconds.\n' % training_time)

    VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

    # get index of predicted dog breed for each image in test set
    VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(VGG16_predictions) == np.argmax(test_targets, axis=1)) / len(
        VGG16_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


def test_VGG19_bottleneck_features(epochs=5, batch_size=20):
    print('testing VGG19 bottleneck features')
    print('loading bottleneck features...')
    start = time()
    bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
    train_files, train_targets = fast_load_dataset('dogImages/train')
    valid_files, valid_targets = fast_load_dataset('dogImages/valid')
    test_files, test_targets = fast_load_dataset('dogImages/test')
    stop = time()
    loading_time = round(stop - start, 0)
    print('bottleneck features loaded in %d seconds.\n' % loading_time)

    print('extracting bottleneck tensors...')
    start = time()
    train_VGG19 = bottleneck_features['train']
    valid_VGG19 = bottleneck_features['valid']
    test_VGG19 = bottleneck_features['test']
    stop = time()
    extraction_time = round(stop - start, 0)
    print('bottleneck tensors extracted in %d seconds.\n' % extraction_time)

    print('defining model...')
    VGG19_model = Sequential()
    VGG19_model.add(Dense(128, activation='relu',
                          input_shape=train_VGG19.shape[1:],
                          kernel_initializer=lecun_normal()))
    VGG19_model.add(Dense(128, activation='relu',
                          kernel_initializer=lecun_normal()))
    VGG19_model.add(GlobalAveragePooling2D())
    VGG19_model.add(Dense(133, activation='softmax',
                          kernel_initializer=lecun_normal()))
    VGG19_model.summary()

    print('compiling model...')
    start = time()
    VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    stop = time()
    compiling_time = round(stop - start, 0)
    print('model compiled in %d seconds.\n' % compiling_time)

    print('training model...')
    start = time()
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5',
                                   verbose=1, save_best_only=True)

    VGG19_model.fit(train_VGG19, train_targets,
                    validation_data=(valid_VGG19, valid_targets),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer, TQDMCallback()], verbose=0)
    stop = time()
    training_time = round(stop - start, 0)
    print('model trained in %d seconds.\n' % training_time)

    VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

    # get index of predicted dog breed for each image in test set
    VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(VGG19_predictions) == np.argmax(test_targets, axis=1)) / len(
        VGG19_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


def test_resnet50_bottleneck_features(epochs=5, batch_size=20, augment_data=False):
    print('testing Resnet50 bottleneck features')
    print('loading bottleneck features...')
    start = time()
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_files, train_targets = fast_load_dataset('dogImages/train')
    valid_files, valid_targets = fast_load_dataset('dogImages/valid')
    test_files, test_targets = fast_load_dataset('dogImages/test')
    stop = time()
    loading_time = round(stop - start, 0)
    print('bottleneck features loaded in %d seconds.\n' % loading_time)

    print('extracting bottleneck tensors...')
    start = time()
    train_resnet50 = bottleneck_features['train']
    valid_resnet50 = bottleneck_features['valid']
    test_resnet50 = bottleneck_features['test']
    stop = time()
    extraction_time = round(stop - start, 0)
    print('bottleneck tensors extracted in %d seconds.\n' % extraction_time)

    print('defining model...')
    resnet50_model = Sequential()
    resnet50_model.add(Flatten(input_shape=train_resnet50.shape[1:], name='extracted_features'))
    resnet50_model.add(Dense(133, name='labels'))
    resnet50_model.add(Activation('softmax', name='softmax'))

    resnet50_model.summary()

    print('compiling model...')
    start = time()
    resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    stop = time()
    compiling_time = round(stop - start, 0)
    print('model compiled in %d seconds.\n' % compiling_time)

    print('training model...')
    start = time()
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                                   verbose=1, save_best_only=True)

    if augment_data:
        pass
        datagen = ImageDataGenerator(rotation_range=45,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     fill_mode='nearest',
                                     horizontal_flip=True,
                                     data_format='channels_last')

        train_generator = datagen.flow(train_resnet50, train_targets, batch_size=batch_size)
        validation_generator = datagen.flow(valid_resnet50, valid_targets, batch_size=batch_size)

        resnet50_model.fit_generator(generator=train_generator,
                                     steps_per_epoch=100,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=[checkpointer],
                                     validation_data=validation_generator,
                                     validation_steps=100)
    else:
        resnet50_model.fit(train_resnet50, train_targets,
                           validation_data=(valid_resnet50, valid_targets),
                           epochs=epochs, batch_size=batch_size,
                           callbacks=[checkpointer, TQDMCallback()], verbose=0)
    stop = time()
    training_time = round(stop - start, 0)
    print('model trained in %d seconds.\n' % training_time)

    resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_resnet50]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(test_targets, axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


def test_InceptionV3_bottleneck_features(epochs=5, batch_size=20):
    print('testing InceptionV3 bottleneck features')
    print('loading bottleneck features...')
    start = time()
    bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
    train_files, train_targets = fast_load_dataset('dogImages/train')
    valid_files, valid_targets = fast_load_dataset('dogImages/valid')
    test_files, test_targets = fast_load_dataset('dogImages/test')
    stop = time()
    loading_time = round(stop - start, 0)
    print('bottleneck features loaded in %d seconds.\n' % loading_time)

    print('extracting bottleneck tensors...')
    start = time()
    train_InceptionV3 = bottleneck_features['train']
    valid_InceptionV3 = bottleneck_features['valid']
    test_InceptionV3 = bottleneck_features['test']
    stop = time()
    extraction_time = round(stop - start, 0)
    print('bottleneck tensors extracted in %d seconds.\n' % extraction_time)

    print('defining model...')
    InceptionV3_model = Sequential()
    InceptionV3_model.add(Dense(128, activation='relu',
                          input_shape=train_InceptionV3.shape[1:],
                          kernel_initializer=lecun_normal()))
    InceptionV3_model.add(Dense(128, activation='relu',
                          kernel_initializer=lecun_normal()))
    InceptionV3_model.add(GlobalAveragePooling2D())
    InceptionV3_model.add(Dense(133, activation='softmax', kernel_initializer=lecun_normal()))

    InceptionV3_model.summary()

    print('compiling model...')
    start = time()
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    stop = time()
    compiling_time = round(stop - start, 0)
    print('model compiled in %d seconds.\n' % compiling_time)

    print('training model...')
    start = time()
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5',
                                   verbose=1, save_best_only=True)

    InceptionV3_model.fit(train_InceptionV3, train_targets,
                          validation_data=(valid_InceptionV3, valid_targets),
                          epochs=epochs, batch_size=batch_size,
                          callbacks=[checkpointer, TQDMCallback()], verbose=0)
    stop = time()
    training_time = round(stop - start, 0)
    print('model trained in %d seconds.\n' % training_time)

    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(test_targets, axis=1)) / len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

#
# Resnet50
#


def test_save_resnet50():
    print("downloading weights...")
    model = ResNet50(include_top=False, weights='imagenet')
    print("weights downloaded!")
    model.save_weights(filepath='model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print("weights saved!")


def test_load_resnet50():
    print("loading weights...")
    model = ResNet50(include_top=False, weights='model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print("weights loaded!")

#
# main method
#


def main():
    print("START")
    start = time()
    # summarize_model(model_from_scratch)
    main_test(model_from_scratch, epochs=1, batch_size=20, resume=True)
    # test_VGG16_bottleneck_features()
    # test_VGG19_bottleneck_features()
    # test_resnet50_bottleneck_features()
    # test_InceptionV3_bottleneck_features()
    stop = time()
    total_time = round(stop - start, 0)
    print("total runtime was %d seconds\n" % total_time)
    print("END")


if __name__ == '__main__':
    main()
