import numpy as np
import tensorflow as tf
# import unittests

def create_training_data():

    # Defining feature and target tensors with values for houses with 1 up to 6 bedrooms
    n_bedrooms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
    price_in_hundreds_of_thousands = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype = float)

    return n_bedrooms, price_in_hundreds_of_thousands

def define_and_compile_model():
    # Defining Model
    model = tf.keras.Sequential([
        
        # Define input with appropriate shapes
        tf.keras.Input(shape=(1,)),

        # Define the Dense layer
        tf.keras.layers.Dense(units=1)
    ])

    # Compiling model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    return model

def train_model():
    # Define feature and target tensors with values for hourses with 1 up to 6 bedrooms
    n_bedrooms, price_in_hundreds_of_thousands = create_training_data()
    model = define_and_compile_model()

    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)

    return model

def main():
    features, targets = create_training_data()
    print(f"Features have shape: {features.shape}")
    print(f"Targets have shape: {targets.shape}")

    # unittests.test_create_training_data(create_training_data)

    untrained_model = define_and_compile_model()

    untrained_model.summary()

    # unittests.test_define_and_compile_model(define_and_compile_model)

    trained_model = train_model()

    new_n_bedrooms = np.array([7.0])
    predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
    print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")

    # unittests.test_trained_model(trained_model)



if __name__ == "__main__":
    main()