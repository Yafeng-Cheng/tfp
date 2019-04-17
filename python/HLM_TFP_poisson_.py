
# coding: utf-8

# In[1]:


import csv
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

import tensorflow as tf
import tensorflow_probability as tfp


# In[2]:


import urllib
import os
os.getcwd()


# In[3]:





inv_scale_transform = lambda y: np.log(y)  # Not using TF here.
fwd_scale_transform = tf.exp



def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.
    
    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)


class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def session_options(enable_gpu_ram_resizing=True, enable_xla=False):
    """
    Allowing the notebook to make use of GPUs if they're available.
    
    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
    algebra that optimizes TensorFlow computations.
    """
    config = tf.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        # `allow_growth=True` makes it possible to connect multiple colabs to your
        # GPU. Otherwise the colab malloc's all GPU ram.
        config.gpu_options.allow_growth = True
    if enable_xla:
        # Enable on XLA. https://www.tensorflow.org/performance/xla/.
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
    return config


def reset_sess(config=None):
    """
    Convenience function to create the TF graph & session or reset them.
    """
    if config is None:
        config = session_options(enable_gpu_ram_resizing=True, enable_xla=False)
    global sess
    tf.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession(config=config)

reset_sess()


# In[23]:


radon = pd.read_csv('test_data/tmp/radon/radon.csv')
radon['radon'] = (np.round(np.exp(radon.radon),1)*10).astype(int)
radon.head()


# In[29]:



tfd = tfp.distributions
dtype = np.float32

log_county_uranium_ppm = radon[
    ['county', 'log_uranium_ppm']].drop_duplicates().values[:, 1]
# log_county_uranium_ppm = log_county_uranium_ppm.astype(dtype)

num_counties = len(log_county_uranium_ppm)
# county = np.int32(radon.county.values)
# floor = dtype(radon.floor.values)
# radon = dtype(radon.radon.values)

# Set the chain's start state.
initial_chain_state = [
    tf.ones([num_counties], dtype=tf.float32, name="init_sigma"),
    tf.ones([num_counties], dtype=tf.float32, name="init_gamma"),
    tf.ones([2], dtype=tf.float32, name="init_beta")
]

unconstraining_bijectors = [
    tfp.bijectors.Exp(),       # Maps a positive real to R.
    tfp.bijectors.Identity(),       # Maps a positive real to R.
    tfp.bijectors.Identity(),   # Maps [0,1] to R.  
]

def joint_log_prob2(
    my_sigma, my_gamma, my_beta, 
    floor = dtype(radon.floor.values), radon = dtype(radon.radon.values), 
    county = np.int32(radon.county.values), log_county_uranium_ppm = log_county_uranium_ppm.astype(dtype)
):
    num_counties = len(log_county_uranium_ppm)
    
    rv_sigma = tfd.Independent(
        tfd.InverseGamma(
            concentration = tf.ones(num_counties, dtype=tf.float32),
            rate = tf.ones(num_counties, dtype=tf.float32)
    ),reinterpreted_batch_ndims=1)
    
    rv_gamma = tfp.distributions.Independent(
        tfp.distributions.Normal(
            loc=tf.zeros(num_counties, dtype=tf.float32),
            scale=my_sigma),
        reinterpreted_batch_ndims=1)
    
    rv_beta = tfp.distributions.Independent(
        tfp.distributions.Normal(
            loc=tf.zeros([2], dtype=tf.float32),
            scale=tf.constant([10,10], dtype=tf.float32)),
        reinterpreted_batch_ndims=1)
    
    fixed_effects = my_beta[0] + my_beta[1] * floor
    
    random_effects = tf.gather(
        my_gamma * log_county_uranium_ppm,
        indices=tf.to_int32(county),
        axis=-1)
    linear_predictor = fixed_effects + random_effects
    
    
    lambda_ = tf.math.exp(linear_predictor)
    rv_radon = tfd.Poisson(rate=lambda_)
    
    return (
        rv_beta.log_prob(my_beta)+\
        rv_sigma.log_prob(my_sigma)+\
        rv_gamma.log_prob(my_gamma)+\
        tf.reduce_sum(rv_radon.log_prob(radon), axis=-1)
    )

def unnormalized_log_posterior(my_sigma, my_gamma, my_beta):
    return joint_log_prob2(
        my_sigma, my_gamma, my_beta, floor, radon, county, log_county_uranium_ppm
    )

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.0001, dtype=tf.float32),#np.array(0.2, dtype=dtype),
        trainable=False,
        use_resource=False
    )


[
    sigma_samples,
    gamma_samples,
    beta_sample
], kernel_results = tfp.mcmc.sample_chain(
    num_results=100,
    num_burnin_steps=20,
    current_state=initial_chain_state,
    kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=joint_log_prob2,
            num_leapfrog_steps=2,
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors))

# Initialize any created variables.
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()


# In[30]:


evaluate(init_g)
evaluate(init_l)
[
    sigma_samples_,
    gamma_samples_,
    beta_sample_,
    kernel_results_
] = evaluate([
    sigma_samples,
    gamma_samples,
    beta_sample,
    kernel_results
])


print("acceptance rate: {}".format(
    kernel_results_.inner_results.is_accepted.mean()))
print("final step size: {}".format(
    kernel_results_.inner_results.extra.step_size_assign[-20:].mean()))


# In[ ]:


# for i in range(28):
#     plt.plot(gamma_samples_[:,i])
#     plt.show()


# In[ ]:


# for i in range(sigma_samples_.shape[1]):
#     plt.figure(figsize=(12.5, 15))
#     #histogram of the samples:

#     ax = plt.subplot(311)
# #     ax.set_autoscaley_on(False)

#     plt.hist(
#         sigma_samples_[:,i], histtype='stepfilled', bins=50, alpha=0.85,
#         label=r"posterior of $\lambda_1$", color=TFColor[0], density=True)
#     plt.legend(loc="upper left")
#     plt.title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
# #     plt.xlim([15, 30])
#     plt.xlabel(r"$\lambda_1$ value")
#     plt.show()


# In[ ]:


# n_sample = len(sigma_samples_[-2000:,0])
# y_hat=[0]*n_sample


# In[ ]:


# radon_ = radon.iloc[0,0]
# floor_ = radon.iloc[0,1]
# county_ = int(radon.iloc[0,2])
# log_uranium_ppm_ = radon.iloc[0,3]

# for i in range(n_sample):
#     lambda_i = beta_sample_[-2000+i, ]
#     lambda_i = lambda_i[0]+lambda_i[1]*floor_
#     lambda_i = lambda_i+ log_uranium_ppm_*np.random.normal(0, gamma_samples_[-2000+i,county_])
    
#     y_hat[i] = np.random.poisson(lam=lambda_i, size = 1)
# plt.hist(y_hat)


# In[ ]:


# gamma_samples_[:,int(county[0])]

