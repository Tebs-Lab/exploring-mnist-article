model = create_dense([32] * 5)
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, epochs=50, validation_split=.1, verbose=True)
model.evaluate(x_test, y_test, verbose=True)
