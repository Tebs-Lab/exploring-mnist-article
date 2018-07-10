def new_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=.1, verbose=True)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=True)
    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

for nodes_per_layer in [32, 128, 512]:
    for layers in [3, 4, 5]:
        model = create_dense([nodes_per_layer] * layers)
        model.summary()
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        for _ in range(5):
            new_evaluate(model, x_train, y_train, x_test, y_test)
