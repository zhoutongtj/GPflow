from typing import Callable, List, Optional, TypeVar, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList, set_callback_parameters

from ..models import BayesianModel
from ..optimizers import get_optimizer


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
            assert isinstance(objective, str) and isinstance(optimizer, str)
            self.model.compile(loss=objective,
                               optimizer=optimizer,
                               metrics=kwargs.get('metrics', None)
                               )

    def fit(self,
            train_data: Data,
            train_labels: Data = None,
            var_list: List[tf.Variable] = None,
            epochs: int = 1,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            validation_data: Data = None,
            batch_size: int = None,
            jit: bool = True
            ):
        if self.is_keras_model:
            self.model.fit(
                x=train_data,
                y=train_labels,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_data
            )
        else:
            self._default_training_loop(
                train_data=train_data,
                train_labels=train_labels,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_data,
                var_list=var_list,
                jit=jit
            )

    def _validate_training_data(self, train_data: Data, train_labels: Data):
        if train_labels is None:
            if isinstance(train_data, Tuple):
                return tf.data.Dataset.from_tensors((train_data[0], train_data[1]))
            elif isinstance(train_data, tf.data.Dataset):
                return train_data
        else:
            return tf.data.Dataset.from_tensors((train_data, train_labels))

    def loss_epoch(self, dataset: tf.data.Dataset):
        X, y = iter(dataset).next()
        return self.objective(X, y)

    def _default_training_loop(self,
                               train_data: Data,
                               train_labels: Data = None,
                               epochs: int = 1,
                               callbacks: List[tf.keras.callbacks.Callback] = None,
                               validation_data: Data = None,
                               var_list: List[tf.Variable] = None,
                               jit: bool = False):
        """
        Simple generic training loop. At each iteration uses a GradientTape to compute
        the gradients of a loss function with respect to a set of variables.
        """
        dataset = self._validate_training_data(train_data, train_labels)
        callbacks = CallbackList(callbacks)
        set_callback_parameters(callbacks, self.model, epochs=epochs,
                                do_validation=validation_data is not None)

        def optimization_step(dataset):
            with tf.GradientTape() as tape:
                tape.watch(var_list)
                loss = self.loss_epoch(dataset=dataset)
                grads = tape.gradient(loss, var_list)
            self.default_optimizer.apply_gradients(zip(grads, var_list))

        if jit:
            optimization_step = tf.function(optimization_step)

        callbacks.on_train_begin()
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            optimization_step(dataset)
            callbacks.on_epoch_end(epoch)
        callbacks.on_train_end()

    def get_objective(self, objective: Union[str, tf.losses.Loss]):
        # TODO (@sergio_pasc): remove this hack that makes all losses work with loss(X, Y)
        if isinstance(objective, str):
            if (isinstance(self.model, BayesianModel) and
                    objective in self.model.all_objectives.keys()):
                return self.model.all_objectives.get(objective)
            else:
                objective = tf.keras.losses.get(objective)
                return lambda x, y: objective(self.model(x), y)
        elif isinstance(objective, tf.losses.Loss):
            return lambda x, y: objective(self.model(x), y)
        else:
            raise NotImplementedError


