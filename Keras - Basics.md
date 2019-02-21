# What is Keras
- High-level deep learning library(neural network API) built over Tensorflow, CNTK, or Theano
- Initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System)
# Dependencies
- Tensorflow: to use tensorflow's backend
- cuDNN: to run Keras on GPU
- HDF5 & h5py: to save Keras models to disk
- graphviz & pydot
- Keras
    -> Should be installed via `pip install ___` 
# Keywords
## Model
- a way to organize layers -> the core data structure of Keras
- ex. `Sequential` model: a linear stack of layers
- ex. `Keras functional API`: arbitrary graphs of layers
## Epoch
- In one Epoch, an entire dataset is passed forward and backward through the nerual network once
- But the dataset is too big to feed to the computer at once -> divide into batches
- Number of Epoch increases -> weights are changed more -> optimal to overfitting
## Batch
- Divided dataset
- Batch Size: Total number of training examples present in a single batch
## Iteration
- Number of batches needed to complete one epoch
- Number of iterations = Number of batches in one epoch
# Example (Sequential model)
## Create model
```python
from keras.models import Sequential
model = Sequential()
```
## Add layers
- by adding them via add() method
```python
from keras.layers import Dense, Activation
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```
- or define them when creating model
```python
from keras.layers import Dense, Activation
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu')])
```
## Define input shape
- Define shape when creating layer
    - `input_shape`: a tuple of integers or `None` entries
    - batch dimension is not included
- Define dimension when creating layer
    - `input_dim`, `input_length`
- Define batch size when creating layer
        - `batch_size`
_`input_shape=(784, None)` = `input_dim=784`_
## Configure learning process
- Optimizer: use predefined optimizer or an instance of the `Optimizer` class
- Loss function: the objectvie that the model will try to minimize. use predefined loss function(`categorical_crossentropy`) or a objective funtion
- Metrics: use predefined metric(`metrics=['accuracy']`) or a custom metric function
## Compile
- Use `compile()` method
```python
model.compile(optimizer='tmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```
## Train
- input data & labels: numpy arrays
- labels: categorical one-hot encoding
    ```python
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
    ```
- Train: iteration on the data in batches
    ```python
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)
    ```
# Source
- [Keras Documentation](https://keras.io/)
- [Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
