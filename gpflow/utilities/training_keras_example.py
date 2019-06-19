import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow import default_float, default_jitter
from gpflow.models import SVGP
from gpflow.utilities.training import TrainingProcedure

tf.keras.backend.set_floatx('float64')

# -----------------------------------------------------
# DATA
# -----------------------------------------------------

np.random.seed(0)
N = 100
num_hidden = 200
Xt = np.random.rand(N, 1) * 10
Yt = np.sin(Xt) + 0.1 * np.random.randn(*Xt.shape)

Xtest = np.random.rand(1000, 1) * 10
Ytest = np.sin(Xtest) + 0.1 * np.random.randn(*Xtest.shape)

dataset = tf.data.Dataset.from_tensors((Xt, Yt))
test_dataset = tf.data.Dataset.from_tensors((Xtest, Ytest))

# -----------------------------------------------------
# Model definition
# -----------------------------------------------------

model_gp = SVGP(gpflow.kernels.RBF(), gpflow.likelihoods.Gaussian(),
                feature=np.linspace(0, 10, 10).reshape(10, 1)
                )


# -----------------------------------------------------
# Callback for metrics
# -----------------------------------------------------

class SampleBasedMSE(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, num_samples=1, log_dir='logs', freq=50):
        super().__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.num_samples = num_samples
        self.freq = freq

        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def generate_samples(self, mu, var, num_samples):
        jitter = tf.eye(tf.shape(mu)[0], dtype=default_float()) * default_jitter()
        samples = [None] * self.model.bayesian_model.num_latent
        for i in range(self.model.bayesian_model.num_latent):
            L = tf.linalg.cholesky(var[i, ...] + jitter)
            shape = tf.stack([L.shape[0], num_samples])
            V = tf.random.normal(shape, dtype=L.dtype)
            samples[i] = mu[:, i:(i + 1)] + L @ V
        return tf.transpose(tf.stack(samples))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            X_val, y_val = self.validation_data
            y_mu, y_var = self.model.bayesian_model.predict_y(X_val)
            # Generate samples of GP model
            y_samples = self.generate_samples(y_mu, y_var, num_samples=self.num_samples)
            # Compute MSE accross inputs and samples
            mse = tf.keras.metrics.mse(y_samples, y_val).numpy().mean()
            # Plot samples
            fig = plt.figure(0)
            for i in range(self.num_samples):
                plt.plot(X_val, y_samples[i, ...], '.')
            image = self.plot_to_image(fig)
            # Write to summary
            with self.summary_writer.as_default():
                tf.summary.scalar('mse_samples', data=mse, step=epoch)
                tf.summary.image('sample', data=image, step=epoch)

    @staticmethod
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


# -----------------------------------------------------
# Training
# -----------------------------------------------------


training = TrainingProcedure(model=model_gp,
                             objective='neg_log_marginal_likelihood',
                             optimizer='adam',
                             metrics=[
                                 'mse',
                                 'log_likelihood'
                             ]
                             )

logdir = '/tmp/logs'

training.fit(
    train_data=Xt,
    train_labels=Yt,
    validation_data=(Xtest, Ytest),
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        SampleBasedMSE((Xtest, Ytest), num_samples=5, log_dir=logdir)
    ],
    epochs=1000
)

# -----------------------------------------------------
# Evaluate / Plot trained model
# -----------------------------------------------------

Y_predict_gp = model_gp.predict_y(Xtest)[0]
# Plot predictions
plt.plot(Xtest, Ytest, 'b.')
plt.plot(Xtest, Y_predict_gp, 'r.')
plt.show()
