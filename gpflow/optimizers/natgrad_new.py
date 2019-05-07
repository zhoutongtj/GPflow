# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import numpy as np
import tensorflow as tf


class ForwardTape(tf.GradientTape):
    def forward_gradient(self, targets, sources, input_gradients=None):
        target_list = targets if isinstance(targets, list) else [targets]
        with tf.GradientTape() as tape:
            v = [tf.Variable(tf.ones_like(target)) for target in target_list]
            g = self.gradient(targets, sources, output_gradients=v)
            forward_gradient = tape.gradient(g, v, output_gradients=input_gradients)
        return forward_gradient

x = tf.Variable(23.)
## FORWARDS
with ForwardTape() as tape:
    f = tf.square(x)
    grad_f_x = tape.forward_gradient(f, [x])

print('grad of f wrt x', grad_f_x[0] )


with ForwardTape() as tape:
    f = tf.square(x)
    loss = tf.math.sin(f)
    grad_l_x = tape.forward_gradient(loss, [f], input_gradients=grad_f_x)

print('grad of l wrt x FORWARDS:', grad_l_x[0] )


## BACKWARDS

with tf.GradientTape() as tape:
    f = tf.square(x)
    loss = tf.math.sin(f)
    grad_l_f = tape.gradient(loss, [f])

with tf.GradientTape() as tape2:
    f = tf.square(x)
    loss = tf.math.sin(f)
    grad_l_x = tape2.gradient(f, [x], output_gradients=grad_l_f)

print('grad of l wrt x BACKWARDS:', grad_l_x[0] )

## AS usual

with tf.GradientTape() as tape:
    f = tf.square(x)
    loss = tf.math.sin(f)
    grad_l_x = tape.gradient(loss, [x])
print('grad of l wrt x sa usual:', grad_l_x[0] )



