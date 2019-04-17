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


import urllib
import os
os.getcwd()


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


def my_split(x, f, return_list = True):
    res = {}
    for v, k in zip(x, f):
        if k in res.keys():
            res[k].append(v)
        else:
            res[k] = [v]
    if return_list:
        res = [res[k] for k in res.keys()]
    return res

import pickle
with open('/projects/bbr/processed_data/engineered_v13_raw.pickle', 'rb') as file:
    df_list = pickle.load(file)
df = df_list['df_raw']

fix_col = [
    'PRICE'
    # , 'gift_pack_price_55', 'gift_pack_price_55_112',
    # 'gift_pack_price_112_163', 'gift_pack_price_163','CPI'
]

random_col = [
    'PRICE_percent'
    # ,'tmax', 'sunshine','QTY_ar_1', 
    # 'inventory_ALL_ar_1', 'PRICE_percent_ar_1'
]


tfd = tfp.distributions
dtype = np.float32

fix_col_mat = df[fix_col].values
random_col_mat = df[random_col].values

num_group = len(df.lineage.unique())

# Set the chain's start state.
initial_chain_state = [
    tf.ones([num_group], dtype=tf.float32, name="init_sigma_1"),
    # tf.ones([num_group], dtype=tf.float32, name="init_sigma_2"),
    # tf.ones([num_group], dtype=tf.float32, name="init_sigma_3"),
    # tf.ones([num_group], dtype=tf.float32, name="init_sigma_4"),
    # tf.ones([num_group], dtype=tf.float32, name="init_sigma_5"),
    # tf.ones([num_group], dtype=tf.float32, name="init_sigma_6"),
    tf.ones([num_group], dtype=tf.float32, name="init_gamma_1"),
    # tf.ones([num_group], dtype=tf.float32, name="init_gamma_2"),
    # tf.ones([num_group], dtype=tf.float32, name="init_gamma_3"),
    # tf.ones([num_group], dtype=tf.float32, name="init_gamma_4"),
    # tf.ones([num_group], dtype=tf.float32, name="init_gamma_5"),
    # tf.ones([num_group], dtype=tf.float32, name="init_gamma_6"),
    tf.ones([len(fix_col)+1], dtype=tf.float32, name="init_beta")
]

unconstraining_bijectors = [
    tfp.bijectors.Exp(),       # Maps a positive real to R.
    # tfp.bijectors.Exp(),       # Maps a positive real to R.
    # tfp.bijectors.Exp(),       # Maps a positive real to R.
    # tfp.bijectors.Exp(),       # Maps a positive real to R.
    # tfp.bijectors.Exp(),       # Maps a positive real to R.
    # tfp.bijectors.Exp(),       # Maps a positive real to R.
    tfp.bijectors.Identity(),       # Maps a positive real to R.
    # tfp.bijectors.Identity(),       # Maps a positive real to R.
    # tfp.bijectors.Identity(),       # Maps a positive real to R.
    # tfp.bijectors.Identity(),       # Maps a positive real to R.
    # tfp.bijectors.Identity(),       # Maps a positive real to R.
    # tfp.bijectors.Identity(),       # Maps a positive real to R.
    tfp.bijectors.Identity(),   # Maps [0,1] to R.  
]

print(unconstraining_bijectors)
print(initial_chain_state)

def joint_log_prob2(
    my_sigma_1,
    # my_sigma_2,
    # my_sigma_3,
    # my_sigma_4,
    # my_sigma_5,
    # my_sigma_6,
    my_gamma_1, 
    # my_gamma_2, 
    # my_gamma_3, 
    # my_gamma_4, 
    # my_gamma_5, 
    # my_gamma_6, 
    my_beta, 
    fix_col = fix_col_mat, random_col = random_col_mat, 
    group = df.lineage.values, target = df.QTY.values
):
    group_unique =  np.unique(group)
    num_group = len(group_unique)
    n_fix = len(fix_col[0,:])+1
    
    my_sigma = [
        my_sigma_1#,
        # my_sigma_2,
        # my_sigma_3,
        # my_sigma_4,
        # my_sigma_5,
        # my_sigma_6
    ]
    
    my_gamma = [
        my_gamma_1#, 
        # my_gamma_2, 
        # my_gamma_3, 
        # my_gamma_4, 
        # my_gamma_5, 
        # my_gamma_6
    ]
    
    rv_random = [0]*len(random_col[0,:])
    for i_col in range(len(rv_random)):
        rv_random[i_col] = [
            tfd.Independent(#sigma
                tfd.InverseGamma(
                    concentration = tf.constant(np.ones(num_group)/10, dtype=tf.float32),
                    rate = tf.constant(np.ones(num_group)*10, dtype=tf.float32)
            ),reinterpreted_batch_ndims=1),
            tfp.distributions.Independent(#gamma
                tfp.distributions.Normal(
                    loc=tf.zeros(num_group, dtype=tf.float32),
                    scale=my_sigma[i_col]),
                reinterpreted_batch_ndims=1)
        ]
    
    rv_beta = tfp.distributions.Independent(
        tfp.distributions.Normal(
            loc=tf.zeros([n_fix], dtype=tf.float32),
            scale=tf.constant([5]*n_fix, dtype=tf.float32)),
        reinterpreted_batch_ndims=1)
    
    fixed_effects = my_beta[0]*1
    for i_col in range(1, n_fix):
        tmp = tf.reduce_sum(my_beta[i_col] * fix_col[:,(i_col-1)])
        fixed_effects = tf.add(fixed_effects, tmp)
    
    random_effects = [0]*len(rv_random)
    for i_col in range(len(rv_random)):
        random_effects[i_col] = tf.gather(
            my_gamma[i_col] * [sum(i_list) for i_list in my_split(random_col[:,i_col], group)],
            indices=tf.to_int32(group_unique),
            axis=-1)
    
    linear_predictor = fixed_effects
    for i_col in range(len(rv_random)):
        linear_predictor =  tf.add(linear_predictor, tf.reduce_sum(random_effects[i_col]))
    
    lambda_ = tf.math.exp(linear_predictor)
    rv_target = tfd.Poisson(rate=lambda_)
    
    posterior = rv_beta.log_prob(my_beta) + tf.reduce_sum(rv_target.log_prob(target), axis=-1)
    for i_col in range(len(rv_random)):
        tmp =  rv_random[i_col][0].log_prob(my_sigma[i_col])
        posterior += tmp
        tmp =  rv_random[i_col][1].log_prob(my_gamma[i_col])
        posterior += tmp
        
    return posterior


with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(
            [0.0000001]*len(df.lineage.unique())+\
            [0.0000001]*len(df.lineage.unique())+\
            [0.00001]*(len(fix_col)+1), dtype=tf.float32),
        trainable=False,
        use_resource=True
    )

    
    
[
    sigma_samples_1,
    # sigma_samples_2,
    # sigma_samples_3,
    # sigma_samples_4,
    # sigma_samples_5,
    # sigma_samples_6,
    gamma_samples_1,
    # gamma_samples_2,
    # gamma_samples_3,
    # gamma_samples_4,
    # gamma_samples_5,
    # gamma_samples_6,
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


# In[147]:


evaluate(init_g)
evaluate(init_l)
[
    sigma_samples_1_,
    # sigma_samples_2_,
    # sigma_samples_3_,
    # sigma_samples_4_,
    # sigma_samples_5_,
    # sigma_samples_6_,
    gamma_samples_1_,
    # gamma_samples_2_,
    # gamma_samples_3_,
    # gamma_samples_4_,
    # gamma_samples_5_,
    # gamma_samples_6_,
    beta_sample_,
    kernel_results_
] = evaluate([
    sigma_samples_1,
    # sigma_samples_2,
    # sigma_samples_3,
    # sigma_samples_4,
    # sigma_samples_5,
    # sigma_samples_6,
    gamma_samples_1,
    # gamma_samples_2,
    # gamma_samples_3,
    # gamma_samples_4,
    # gamma_samples_5,
    # gamma_samples_6,
    beta_sample,
    kernel_results
])

    
print("acceptance rate: {}".format(
    kernel_results_.inner_results.is_accepted.mean()))
print("final step size: {}".format(
    kernel_results_.inner_results.extra.step_size_assign[-2000:].mean()))

xx = [
    sigma_samples_1_,
    # sigma_samples_2_,
    # sigma_samples_3_,
    # sigma_samples_4_,
    # sigma_samples_5_,
    # sigma_samples_6_,
    gamma_samples_1_,
    # gamma_samples_2_,
    # gamma_samples_3_,
    # gamma_samples_4_,
    # gamma_samples_5_,
    # gamma_samples_6_,
    beta_sample_,
    kernel_results_
]

import pickle
with open("/projects/bbr/code_yc/tfp/python/tfp_poisson.pickle",'wb') as file:
    pickle.dump(
        file
    )
# n_sample = len(sigma_samples_[-2000:,0])
# y_hat=[0]*n_sample
# 
# 
# # In[165]:
# 
# 
# radon_ = radon.iloc[0,0]
# floor_ = radon.iloc[0,1]
# county_ = int(radon.iloc[0,2])
# log_uranium_ppm_ = radon.iloc[0,3]
# 
# for i in range(n_sample):
#     lambda_i = beta_sample_[-2000+i, ]
#     lambda_i = lambda_i[0]+lambda_i[1]*floor_
#     lambda_i = lambda_i+ log_uranium_ppm_*np.random.normal(0, gamma_samples_[-2000+i,county_])
#     
#     y_hat[i] = np.random.poisson(lam=lambda_i, size = 1)
# plt.hist(y_hat)
# 
# 
# # In[159]:
# 
# 
# gamma_samples_[:,int(county[0])]

