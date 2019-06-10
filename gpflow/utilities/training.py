from typing import Callable, List, Optional, TypeVar, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList, set_callback_parameters

from ..models import BayesianModel
from ..optimizers import get_optimizer

tf.keras.backend.set_floatx('float64')


def set_trainable(model: tf.Module, flag: bool = False):
    for variable in model.trainable_variables:
        variable._trainable = flag


def training_loop(closure: Callable[..., tf.Tensor],
                  optimizer: Optional[tf.optimizers.Optimizer] = None,
                  var_list: List[tf.Variable] = None,
                  maxiter=1e3,
                  jit=False):
    """
    Simple generic training loop. At each iteration uses a GradientTape to compute
    the gradients of a loss function with respect to a set of variables.

    :param closure: Callable that constructs a loss function based on data and model being trained
    :param optimizer: tf.optimizers or tf.keras.optimizers that updates variables by applying the
        corresponding loss gradients. Adam is a default optimizer with default settings.
    :param var_list: List of model variables to be learnt during training
    :param maxiter: Maximum number of
    :return:
    """

    optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

    def optimization_step():
        with tf.GradientTape() as tape:
            tape.watch(var_list)
            loss = closure()
            grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if jit:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()


Data = TypeVar('Data', tf.Tensor, np.ndarray, tf.data.Dataset,
               Tuple[tf.Tensor, tf.Tensor], Tuple[np.ndarray, np.ndarray]
               )


class BayesianModelKeras(tf.keras.Model):
    def __init__(self, model: BayesianModel):
        super(BayesianModelKeras, self).__init__()
        self.model = model
        self.model_weights = model.variables

    def call(self, inputs):
        X, Y = inputs[0], inputs[1]
        return self.model.neg_log_marginal_likelihood(X, Y)



class IdentityLoss(tf.keras.losses.Loss):

    def call(self, _, loss):
        return loss


class TrainingProcedure:
    def __init__(self,
                 model: Union[tf.Module, tf.keras.models.Model],
                 objective: Union[str, tf.losses.Loss],
                 optimizer: Union[str, tf.optimizers.Optimizer] = None,
                 **kwargs):
        """
        :param model:
        :param optimizer:
        :param objective:
        :param kwargs: Extra parameters for keras ...
        """
        self.is_keras_model = isinstance(model, tf.keras.Model)
        self.model = model
        self.objective = self.get_objective(objective)
        self.default_optimizer = get_optimizer(optimizer)

        if self.is_keras_model:
            self.model_keras = self.model
        elif isinstance(self.model, BayesianModel):
            self.model_keras = BayesianModelKeras(self.model)

        self.model_keras.compile(loss=objective,
                                 optimizer=optimizer,
                                 metrics=kwargs.get('metrics', None)
                                 )

    def fit(self,
            train_data: Data,
            train_labels: Data = None,
            epochs: int = 1,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            validation_data: Data = None,
            batch_size: int = None,
            ):
        if self.is_keras_model:
            self.model_keras.fit(
                x=train_data,
                y=train_labels,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_data
            )
        elif isinstance(self.model, BayesianModel):
            self.model_keras.fit(
                x=(train_data, train_labels),
                y=np.zeros_like(train_labels),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_data
            )

    def _validate_training_data(self, train_data: Data, train_labels: Data):
        if train_labels is None:
            if isinstance(train_data, Tuple):
                return tf.data.Dataset.from_tensors((train_data[0], train_data[1]))
            elif isinstance(train_data, tf.data.Dataset):
                return train_data
        else:
            return tf.data.Dataset.from_tensors((train_data, train_labels))

    def get_objective(self, objective: Union[str, tf.losses.Loss]):
        # TODO (@sergio_pasc): remove this hack that makes all losses work with loss(X, Y)
        if isinstance(objective, str):
            if isinstance(self.model, BayesianModel):
                return lambda y_true, loss: IdentityLoss()(y_true, loss)
            else:
                objective = tf.keras.losses.get(objective)
                return lambda y_true, x: objective(y_true, self.model(x))
        elif isinstance(objective, tf.losses.Loss):
            return lambda y_true, x: objective(y_true, self.model(x))
        else:
            raise NotImplementedError
