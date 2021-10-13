import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sympy import *
import math
import sympy as sp
from itertools import combinations


varia_dict = {
    'age':'t',
    'MAT': 'T',
    'precipitation':'P',
    'erosion':'E',
    'quartz':'Q',
    'albite':'A'
}

### PARAMETEER LIST

variables = ['age','MAT','precipitation','quartz']


lr = 0.001
NUM_VARIABLE = len(variables)


def formula(depth,A,gk):

    return 1 / (1 + A * tf.math.exp(gk * depth))

def tt_module(Z,hsize=[16, 8]):
    ### set 0-3
    # h1 = tf.layers.dense(Z,hsize[0])
    h2 = tf.layers.dense(Z, 16, activation='sigmoid')
    # out = tf.layers.dense(h2,8,activation='sigmoid')
    out = tf.layers.dense(h2,1,activation='linear')
    return out

def analytical(depth, A, tt_output):
    return 1 / (1 + A * tf.math.exp(tt_output * depth))
#
#
depth = tf.placeholder(tf.float32, [None, 1])  ## depth
Z = tf.placeholder(tf.float32, [None, NUM_VARIABLE])  ## Age, MAT, precipitation,erosion,QUARTZ
A = tf.placeholder(tf.float32, [None, 1])  ##
concentration = tf.placeholder(tf.float32, [None, 1])
gk = tf.placeholder(tf.float32, [None, 1])


tt_output = tt_module(Z)
concentration_pred = analytical(depth, A, tt_output)
phy_pred = formula(depth, A, gk)
loss = tf.reduce_mean(tf.squared_difference(concentration_pred, concentration))
phy_loss = tf.reduce_mean(tf.squared_difference(phy_pred, concentration))
step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # G Train step

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
saver = tf.train.Saver()

temperature = Symbol('T')
age = Symbol('t')
precipitation = Symbol('P')
erosion = Symbol('E')
quartz = Symbol('Q')


saver.restore(sess, "./results/iter_43a/model.ckpt")


print("Model restored.")

input_mat = np.array([age,temperature,precipitation,quartz])

for i in range(int(len(tf.trainable_variables())/2)):
  w = tf.trainable_variables()[2*i].eval(sess)
  b = tf.trainable_variables()[2*i+1].eval(sess)
  input_mat = np.dot(input_mat,w) + b
  if i == 0 :
      input_mat = [1 / (sp.exp(-a) + 1) for a in input_mat]

print(input_mat)
# print(simplify(input_mat))