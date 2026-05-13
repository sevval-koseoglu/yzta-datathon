#create a neural network model using keras
from tensorflow import keras

def create_model(input_shape):
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        
        
        keras.layers.Dense(32),
        keras.layers.LeakyReLU(negative_slope=0.01),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(1)   
        
    ])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        min_delta=1e-5,
        cooldown=2,
        verbose=1,
        mode="min"
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    return model, [reduce_lr, early_stop]
