import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
import matplotlib.pyplot as plt

import gpflow
from gpflow.utilities.training import TrainingProcedure, KerasBayesianModel
from gpflow.models import BayesianModel, SVGP

## DATA


np.random.seed(0)
N = 100
num_hidden = 200
Xt = np.random.rand(N, 1) * 10
Yt = np.sin(Xt) + 0.1 * np.random.randn(*Xt.shape)

Xtest = np.random.rand(1000, 1) * 10
Ytest = np.sin(Xtest) + 0.1 * np.random.randn(*Xtest.shape)

dataset = tf.data.Dataset.from_tensors((Xt, Yt))
test_dataset = tf.data.Dataset.from_tensors((Xtest, Ytest))
# plt.plot(Xt, Yt, 'r.')


## KERAS HELPERS

tf.keras.backend.set_floatx('float64')


class IdentityLoss(tf.keras.losses.Loss):

    def call(self, _, loss):
        return loss


## MODELS

def run_keras_fit():
    model_gp = SVGP(gpflow.kernels.RBF(), gpflow.likelihoods.Gaussian(),
                    feature=np.linspace(0, 10, 10).reshape(10, 1))

    class Metrics(tf.keras.callbacks.Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data

        def on_train_begin(self, logs=None):
            self._data = []

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val = self.validation_data
            if epoch % 100 == 0:
                Y_predict_gp = self.model.bayesian_model.predict_y(X_val)[0]
                plt.plot(Xtest, Ytest, 'b.')
                plt.plot(Xtest, Y_predict_gp, 'r.')
                plt.show()

    callbacks = [Metrics((Xtest, Ytest))]

    training = TrainingProcedure(model=model_gp,
                                 objective='neg_log_marginal_likelihood',
                                 optimizer='adam',
                                 metrics=[
                                     # 'mse',
                                     # 'neg_log_marginal_likelihood',
                                     # 'log_likelihood'
                                 ]
                                 )
    t0 = time.time()
    training.fit(
        train_data=Xt,
        train_labels=Yt,
        validation_data=(Xtest, Ytest),
        callbacks=callbacks,
        epochs=1000
    )
    t_f = time.time()
    Y_predict_gp = model_gp.predict_y(Xtest)[0]
    mse_test = tf.keras.losses.mse(Y_predict_gp.numpy().T, Ytest.T)
    print('{} secs to train tf.Module model with MSE {} in test set'.format(t_f - t0, mse_test))
    # Plot predictions
    plt.plot(Xtest, Ytest, 'b.')
    plt.plot(Xtest, Y_predict_gp, 'r.')
    plt.show()


if __name__ == '__main__':
    run_keras_fit()
    # model_gp = SVGP(gpflow.kernels.RBF(), gpflow.likelihoods.Gaussian(),
    #                 feature=np.linspace(0, 10, 10).reshape(10, 1))
    # model_keras = KerasBayesianModel(model=model_gp, objective_name='neg_log_marginal_likelihood',
    #                    inference_name='predict_y')
    # model_keras.compile(
    #     loss=None,
    #     optimizer='adam'
    # )
    # data = tf.data.Dataset.from_tensors((Xt, Yt))
    # data_labels = data.map(lambda x, y: y)
    # data = tf.data.Dataset.zip((data, data_labels))
    # it = iter(data)
    # input, output = next(it)
    # output_pred = model_keras(input)
    # # model_keras = tf.keras.Model(inputs=input, outputs=output_pred)
    # # model_keras.compile(
    # #     loss=None,
    # #     optimizer='adam'
    # # )
    # print(model_keras.summary())
