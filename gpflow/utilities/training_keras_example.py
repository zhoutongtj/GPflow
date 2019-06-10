import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

import gpflow
from gpflow.utilities.training import TrainingProcedure
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
    loss_neg_loglik = IdentityLoss()

    training = TrainingProcedure(model=model_gp, objective=loss_neg_loglik, optimizer='adam')

    training.fit(
        train_data=Xt,
        train_labels=Yt,
        epochs=1000
    )


if __name__ == '__main__':
    run_keras_fit()
