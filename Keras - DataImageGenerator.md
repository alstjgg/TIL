# Keras
## ImageDataGenerator
- ImageDataGenerator [documentation](https://keras.io/preprocessing/image/) from Keras
- [Source Code](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py)
- Generate batches of tensor image data with real-time data augmentation
- Data will be looped over in batches
### ImageDataGenerator class
```python
keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest', cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None)z
```
 `apply_transform`
- Applies a transformation to an image according to given parameters

`fit`
- Fits the data generator to some sample data

`flow`
- Takes data & label arrays, generates batches of augmented data 

`flow_from_dataframe`
- Takes the dataframe and the path to a directory and generates batches of augmented/normalized data

`flow_from_directory`
- Takes the path to a directory & generates batches of augmented data

`get_random_transform`
- Generates random parameters for a transformation

`random_transform`
- Applies a random transformation to an image

`standardize`
- Applies the normalization configuration to a batch of inputs
### ImageDataGenerator methods
- `apply_transform(x, transform_parameters)`
    - `x`: 3D tensor(single image)
    - `transform_parameters`: Dictionary with string-parameter pairs describing transformation
- `fit(x, qugment=False, rounds=1, seed=None)`
    - Fit the data generator to sample data
- `flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)`
    - input: data & labels
    - output: batches of augmented data -> an `Iterator` yielding tuple of (x, y)
    - `x`: Input data. Numpy array or tuple(with first element being an image)
    - `y`: Labels
    ```python
    datagen = ImageDataGenerator()
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / batch_size, epochs=epochs)
    ```
- `flow_from_dataframe(dataframe, directory=None, x_col='filename', y_col='class', target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', drop_duplicates=True)`
    - `dataframe`: Pandas dataframe containing the filepaths of the images
    - `directory`: Path to the directory to read images from
    - return `DataFrameIterator`
- `flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')`
    - `directory`: Path to the target directory. Contains one subdirectory per class
    - `target_size`: Dimensions to which all images will be resized
    - `color_mode`: grayscale, rgb, or rgba
    - return `DirectoryIterator`
    ```python
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory('data/train', target_size=(150, 150), batch_size=32, class_mode='binary')
    test_datagen = ImageDataGenerator()
    validation_generator = test_datagen.flow_from_directory('data/validation', target_size=(150, 150), batch_size=32, class_mode='binary')
    model.fir_generator(train_generator, steps_per_epoch=2000, epochs=50, validation_data=validation_generator, validation_steps=800)
    ```
