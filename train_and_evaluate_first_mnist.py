# sgd is short for stochastic gradient descent
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Note the batch_size and epochs
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=True,
                    validation_split=.1)

# The Keras library reports the value of the value of the loss function
# and the accuracy metric (because we requested it on line 2)
loss, accuracy  = model.evaluate(x_test,
                                 y_test,
                                 verbose=False)

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')
