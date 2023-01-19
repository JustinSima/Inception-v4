""" Compile model for training."""
import tensorflow as tf

def compile_model(model):
    """ Compiles model for training in place."""
    # Categorical cross entropy with label smoothing.
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Exponential weight decay.
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.045,
        decay_steps=2,
        decay_rate=0.94
    )

    # Root mean square propogation optimizer.
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_schedule,
        rho=0.9,
        epsilon=1,
        clipvalue=2
    )

    # Compile model in place.
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[loss]
    )
