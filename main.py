from matplotlib import pyplot
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from random import randint

# Preparing the dataset
# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = 784 # 28 x 28

x_train = x_train.reshape(x_train.shape[0], image_size) # Transform from matrix to vector
x_train = x_train.astype('float32')
x_train /= 255 # Normalize inputs from 0-255 to 0.0-1.0

x_test = x_test.reshape(x_test.shape[0], image_size) # Transform from matrix to vector
x_test = x_test.astype('float32')
x_test /= 255 # Normalize inputs from 0-255 to 0.0-1.0

print('Number of train examples:', x_train.shape[0])
print('Number of test examples:', x_test.shape[0])

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Define input layer
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(image_size,))) # Input layer
model.add(Dense(units=num_classes, activation='softmax')) # Output layer
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 128
epochs = 5

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=True,
                    validation_split=.1)

loss, accuracy  = model.evaluate(x_test,
                                 y_test,
                                 verbose=False)
print()
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')


model_three = Sequential()
model_three.add(Dense(units=128, activation='relu', input_shape=(image_size,)))
model_three.add(Dense(units=128, activation='relu'))
model_three.add(Dense(units=64, activation='relu'))
model_three.add(Dense(units=64, activation='relu'))
model_three.add(Dense(units=num_classes, activation='softmax'))
evaluate(model_three, epochs=20)

model_four = Sequential()
model_four.add(Dense(units=10, activation='relu', input_shape=(image_size,)))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=10, activation='relu'))
model_four.add(Dense(units=num_classes, activation='softmax'))
evaluate(model_four, epochs=20)


model_four = Sequential()
model_four.add(Dense(units=5, activation='relu', input_shape=(image_size,)))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=5, activation='relu'))
model_four.add(Dense(units=num_classes, activation='softmax'))
evaluate(model_four, epochs=20)
