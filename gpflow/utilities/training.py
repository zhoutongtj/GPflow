from typing import Callable, List, Optional, TypeVar, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList, set_callback_parameters

from .defaults import default_float
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


class KerasBayesianModel(tf.keras.Model):
    def __init__(self, model: BayesianModel, objective_name: str, metrics: List[str]):
        super(KerasBayesianModel, self).__init__()
        self.bayesian_model = model
        self.bayesian_model_weights = model.variables
        self.objective_fn = model.all_objectives[objective_name]
        self.metrics_fn = {
            metric_name: model.all_objectives[metric_name] for metric_name in metrics
        }

    def call(self, inputs):
        X, Y = inputs[0], inputs[1]
        self.add_loss(self.objective_fn(X, Y))
        for metric_name, metric_fn in self.metrics_fn.items():
            self.add_metric(metric_fn(X, Y), name=metric_name, aggregation='mean')
        y_pred_mean = self.bayesian_model.predict_y(X)[0]
        return y_pred_mean


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
        self.model = model
        self.objective = self.get_objective(objective)
        self.metrics = self.get_metrics(kwargs.get('metrics', None))
        self.default_optimizer = get_optimizer(optimizer)

        self.model_keras = self.get_keras_model(model)
        self.model_keras.compile(
            loss=self.objective,
            optimizer=self.default_optimizer,
            metrics=self.metrics
        )

    def fit(self,
            train_data: Data,
            train_labels: Data = None,
            epochs: int = 1,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            validation_data: Data = None,
            batch_size: int = None,
            ):
        dataset = self.prepare_dataset(train_data, train_labels)
        validation_dataset = self.prepare_dataset(validation_data)
        self.model_keras.fit(dataset,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks,
                             validation_data=validation_dataset
                             )

    def prepare_dataset(self, train_data: Data, train_labels: Optional[Data]=None):
        if train_labels is None:
            if isinstance(train_data, Tuple):
                data = tf.data.Dataset.from_tensors((train_data[0], train_data[1]))
            elif isinstance(train_data, tf.data.Dataset):
                data = train_data
            else:
                raise NotImplementedError
        else:
            data = tf.data.Dataset.from_tensors((train_data, train_labels))

        if self.objective is self.zero_objective:
            data_labels = data.map(lambda x, y: y)
            data = tf.data.Dataset.zip((data, data_labels))
        return data

    def get_metrics(self, metrics: List):
        metrics = [] if metrics is None else metrics
        if isinstance(self.model, BayesianModel):
            self.model_metrics = list(set(metrics).intersection(self.model.all_objectives))
            return list(set(metrics).difference(self.model.all_objectives))
        return metrics

    def get_objective(self, objective: Union[str, tf.losses.Loss]):
        if isinstance(self.model, BayesianModel):
            if objective in self.model.all_objectives:
                self.objective_name = objective
                return self.zero_objective
            return objective
        return objective

    def get_keras_model(self, model):
        if isinstance(model, tf.keras.Model):
            return model
        elif isinstance(model, BayesianModel):
            return KerasBayesianModel(model, self.objective_name, self.model_metrics)
        else:
            raise NotImplementedError

    @staticmethod
    def zero_objective(*args):
        return tf.zeros((1,), dtype=default_float())
