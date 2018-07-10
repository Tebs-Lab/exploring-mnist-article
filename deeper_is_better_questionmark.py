for layers in range(1, 5):
    model = create_dense([32] * layers)
    evaluate(model, x_train, y_train, x_test, y_test)
