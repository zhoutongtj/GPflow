from gpflow.util import Dispatcher

expectation_dispatcher = Dispatcher('expectation')
quadrature_expectation_dispatcher = Dispatcher('quadrature_expectation')
variational_expectation_dispatcher = Dispatcher('variational_expectation')
