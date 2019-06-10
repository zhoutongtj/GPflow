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


class BayesianModelKeras(tf.keras.Model):
    def __init__(self, model: BayesianModel):
        super(BayesianModelKeras, self).__init__()
        self.model = model
        # Registering the weights
        self.model_weights = model.variables

    def call(self, inputs):
        return self.model.predict_f(inputs, full_cov=False, full_output_cov=False)


class NegLogMarginalLikelihoodSVGP(tf.keras.losses.Loss):
    def __init__(self, model: BayesianModel,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='neg_loglik',
                 **kwargs
                 ):
        super(NegLogMarginalLikelihoodSVGP, self).__init__(reduction=reduction, name=name)
        self.model = model

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        kl = self.model.prior_kl()
        # Here we replace self.model.predict_f(X) with y_pred = BayesianModelKeras.call(X)
        f_mean, f_var = y_pred[0], y_pred[1]
        var_exp = self.model.likelihood.variational_expectations(f_mean, f_var, y_true)
        if self.model.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(y_true.shape[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl


## MODELS

def run_keras_fit():
    model_gp = SVGP(gpflow.kernels.RBF(), gpflow.likelihoods.Gaussian(),
                    feature=np.linspace(0, 10, 10).reshape(10, 1))
    model_keras_gp = BayesianModelKeras(model_gp)
    loss_neg_loglik = NegLogMarginalLikelihoodSVGP(model_gp)

    training = TrainingProcedure(model=model_keras_gp, objective=loss_neg_loglik, optimizer='adam')

    training.fit(
        train_data=Xt,
        train_labels=Yt,
        validation_data=(Xtest, Ytest),
        epochs=1000,
        jit=True
    )


if __name__ == '__main__':
    run_keras_fit()
