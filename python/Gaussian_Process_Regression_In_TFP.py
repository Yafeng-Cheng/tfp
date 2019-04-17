
# coding: utf-8

# In this colab, we explore Gaussian process regression using
# TensorFlow and TensorFlow Probability. We generate some noisy observations from
# some known functions and fit GP models to those data. We then sample from the GP
# posterior and plot the sampled function values over grids in their domains.
# 

# ## Background
# Let $\mathcal{X}$ be any set. A *Gaussian process*
# (GP) is a collection of random variables indexed by $\mathcal{X}$ such that if
# $\{X_1, \ldots, X_n\} \subset \mathcal{X}$ is any finite subset, the marginal density
# $p(X_1 = x_1, \ldots, X_n = x_n)$ is multivariate Gaussian. Any Gaussian
# distribution is completely specified by its first and second central moments
# (mean and covariance), and GP's are no exception. We can specify a GP completely
# in terms of its mean function $\mu : \mathcal{X} \to \mathbb{R}$ and covariance function
# $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$. Most of the expressive power of GP's is encapsulated
# in the choice of covariance function. For various reasons, the covariance
# function is also referred to as a *kernel function*. It is required only to be
# symmetric and positive-definite (see [Ch. 4 of Rasmussen & Williams](
# http://www.gaussianprocess.org/gpml/chapters/RW4.pdf)). Below we make use of the
# ExponentiatedQuadratic covariance kernel. Its form is
# 
# $$
# k(x, x') := \sigma^2 \exp \left( \frac{\|x - x'\|^2}{\lambda^2} \right)
# $$
# 
# where $\sigma^2$ is called the 'amplitude' and $\lambda$ the *length scale*.
# The kernel parameters can be selected via a maximum likelihood optimization
# procedure.
# 
# A full sample from a GP comprises a real-valued function over the entire space
# $\mathcal{X}$ and is in practice impractical to realize; often one chooses a set of
# points at which to observe a sample and draws function values at these points.
# This is achieved by sampling from an appropriate (finite-dimensional)
# multi-variate Gaussian.
# 
# Note that, according to the above definition, any finite-dimensional
# multivariate Gaussian distribution is also a Gaussian process. Usually, when
# one refers to a GP, it is implicit that the index set is some $\mathbb{R}^n$
# and we will indeed make this assumption here.
# 
# A common application of Gaussian processes in machine learning is Gaussian
# process regression. The idea is that we wish to estimate an unknown function
# given noisy observations $\{y_1, \ldots, y_N\}$ of the function at a finite
# number of points $\{x_1, \ldots x_N\}.$ We imagine a generative process
# 
# $$
# \begin{align}
# f \sim \: & \textsf{GaussianProcess}\left(
#     \text{mean_fn}=\mu(x),
#     \text{covariance_fn}=k(x, x')\right) \\
# y_i \sim \: & \textsf{Normal}\left(
#     \text{loc}=f(x_i),
#     \text{scale}=\sigma\right), i = 1, \ldots, N
# \end{align}
# $$
# 
# As noted above, the sampled function is impossible to compute, since we would
# require its value at an infinite number of points. Instead, one considers a
# finite sample from a multivariate Gaussian.
# 
# $$
#   \begin{gather}
#     \begin{bmatrix}
#       f(x_1) \\
#       \vdots \\
#       f(x_N)
#     \end{bmatrix}
#     \sim
#     \textsf{MultivariateNormal} \left( \:
#       \text{loc}=
#       \begin{bmatrix}
#         \mu(x_1) \\
#         \vdots \\
#         \mu(x_N)
#       \end{bmatrix} \:,\:
#       \text{scale}=
#       \begin{bmatrix}
#         k(x_1, x_1) & \cdots & k(x_1, x_N) \\
#         \vdots & \ddots & \vdots \\
#         k(x_N, x_1) & \cdots & k(x_N, x_N) \\
#       \end{bmatrix}^{1/2}
#     \: \right)
#   \end{gather} \\
#   y_i \sim \textsf{Normal} \left(
#       \text{loc}=f(x_i),
#       \text{scale}=\sigma
#   \right)
# $$
# 
# Note the exponent $\frac{1}{2}$ on the covariance matrix: this denotes a
# Cholesky decomposition. Comptuing the Cholesky is necessary because the MVN is
# a location-scale family distribution. Unfortunately the Cholesky decomposition
# is computationally expensive, taking $O(N^3)$ time and $O(N^2)$ space. Much of
# the GP literature is focused on dealing with this seemingly innocuous little
# exponent.
# 
# It is common to take the prior mean function to be constant, often zero. Also,
# some notational conventions are convenient. One often writes $\mathbf{f}$ for the
# finite vector of sampled function values. A number of interesting notations are
# used for the covariance matrix resulting from the application of $k$ to pairs of
# inputs. Following [(Quiñonero-Candela, 2005)][QuinoneroCandela2005], we note
# that the components of the matrix are covariances of function values at
# particular input points. Thus we can denote the covariance matrix as $K_{AB}$
# where $A$ and $B$ are some indicators of the collection of function values along
# the given matrix dimensions.
# 
# [QuinoneroCandela2005]: http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
# 
# For example, given observed data $(\mathbf{x}, \mathbf{y})$ with implied latent function
# values $\mathbf{f}$, we can write
# 
# $$
# K_{\mathbf{f},\mathbf{f}} = \begin{bmatrix}
#   k(x_1, x_1) & \cdots & k(x_1, x_N) \\
#   \vdots & \ddots & \vdots \\
#   k(x_N, x_1) & \cdots & k(x_N, x_N) \\
# \end{bmatrix}
# $$
# 
# Similarly, we can mix sets of inputs, as in
# 
# $$
# K_{\mathbf{f},*} = \begin{bmatrix}
#   k(x_1, x^*_1) & \cdots & k(x_1, x^*_T) \\
#   \vdots & \ddots & \vdots \\
#   k(x_N, x^*_1) & \cdots & k(x_N, x^*_T) \\
# \end{bmatrix}
# $$
# 
# where we suppose there are $N$ training inputs, and $T$ test inputs. The above
# generative process may then be written compactly as
# 
# $$
# \begin{align}
# \mathbf{f} \sim \: & \textsf{MultivariateNormal} \left(
#         \text{loc}=\mathbf{0},
#         \text{scale}=K_{\mathbf{f},\mathbf{f}}^{1/2}
#     \right) \\
# y_i \sim \: & \textsf{Normal} \left(
#     \text{loc}=f_i,
#     \text{scale}=\sigma \right), i = 1, \ldots, N
# \end{align}
# $$
# 
# The sampling operation in the first line yields a finite set of $N$ function
# values from a multivariate Gaussian -- *not an entire function as in the above
# GP draw notation*. The second line describes a collection of $N$ draws from
# *univariate* Gaussians centered at the various function values, with fixed
# observation noise $\sigma^2$.
# 
# With the above generative model in place, we can proceed to consider the
# posterior inference problem. This yields a posterior distribution over function
# values at a new set of test points, conditioned on the observed noisy data from
# the process above.
# 
# With the above notation in place, we can compactly write the posterior
# predictive distribution over future (noisy) observations conditional on
# corresponding inputs and training data as follows (for more details, see §2.2 of
# [Rasmussen & Williams](http://www.gaussianprocess.org/gpml/)). 
# 
# $$
# \mathbf{y}^* \mid \mathbf{x}^*, \mathbf{x}, \mathbf{y} \sim \textsf{Normal} \left(
#     \text{loc}=\mathbf{\mu}^*,
#     \text{scale}=(\Sigma^*)^{1/2}
# \right),
# $$
# 
# where
# 
# $$
# \mathbf{\mu}^* = K_{*,\mathbf{f}}\left(K_{\mathbf{f},\mathbf{f}} + \sigma^2 I \right)^{-1} \mathbf{y}
# $$
# 
# and
# 
# $$
# \Sigma^* = K_{*,*} - K_{*,\mathbf{f}}
#     \left(K_{\mathbf{f},\mathbf{f}} + \sigma^2 I \right)^{-1} K_{\mathbf{f},*}
# $$

# In[ ]:


# !pip install -q tensorflow-probability


# ## Imports

# In[10]:


import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk

get_ipython().magic('pylab inline')
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
get_ipython().magic("config InlineBackend.figure_format = 'png'")


# ## Example: Exact GP Regression on Noisy Sinusoidal Data
# Here we generate training data from a noisy sinusoid, then sample a bunch of
# curves from the posterior of the GP regression model. We use
# [Adam](https://arxiv.org/abs/1412.6980) to optimize the kernel hyperparameters
# (we minimize the negative log likelihood of the data under the prior). We
# plot the training curve, followed by the true function and the posterior
# samples.

# In[11]:


def reset_session():
    """Creates a new global, interactive session in Graph-mode."""
    global sess
    try:
        tf.reset_default_graph()
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession()

reset_session()


# In[28]:


def sinusoid(x):
    return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance, shift = 0):
    """Generate noisy sinusoidal observations at a random set of points.

    Returns:
        observation_index_points, observations
    """
    index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    # y = f(x) + noise
    observations_ = (sinusoid(index_points_+shift) +
                    np.random.normal(
                        loc=0,
                        scale=np.sqrt(observation_noise_variance),
                        size=(num_training_points)))
    return index_points_, observations_


# In[30]:


# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 100
observation_index_points_, observations_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.1)
observation_index_points_1, observations_1 = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.1, shift =2)


# In[31]:


# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.
amplitude = (np.finfo(np.float64).tiny +
             tf.nn.softplus(tf.Variable(initial_value=1.,
                                        name='amplitude',
                                        dtype=np.float64)))
length_scale = (np.finfo(np.float64).tiny +
                tf.nn.softplus(tf.Variable(initial_value=1.,
                                           name='length_scale',
                                           dtype=np.float64)))

observation_noise_variance = (
    np.finfo(np.float64).tiny +
    tf.nn.softplus(tf.Variable(initial_value=1e-6,
                               name='observation_noise_variance',
                               dtype=np.float64)))


# In[32]:


# Create the covariance kernel, which will be shared between the prior (which we
# use for maximum likelihood training) and the posterior (which we use for
# posterior predictive sampling)
kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)


# In[33]:


# Create the GP prior distribution, which we will use to train the model
# parameters.
gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=observation_index_points_,
    observation_noise_variance=observation_noise_variance)

# This lets us compute the log likelihood of the observed data. Then we can
# maximize this quantity to find optimal model parameters.
log_likelihood = gp.log_prob(observations_)# + gp.log_prob(observations_1)


# In[34]:


# Define the optimization ops for maximizing likelihood (minimizing neg
# log-likelihood!)
optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(-log_likelihood)


# In[35]:


get_ipython().run_cell_magic('time', '', "# Now we optimize the model parameters.\nnum_iters = 1000\n# Store the likelihood values during training, so we can plot the progress\nlls_ = np.zeros(num_iters, np.float64)\nsess.run(tf.global_variables_initializer())\nfor i in range(num_iters):\n    _, lls_[i] = sess.run([train_op, log_likelihood])\n\n[\n    amplitude_,\n    length_scale_,\n    observation_noise_variance_\n] = sess.run([\n    amplitude,\n    length_scale,\n    observation_noise_variance])\nprint('Trained parameters:'.format(amplitude_))\nprint('amplitude: {}'.format(amplitude_))\nprint('length_scale: {}'.format(length_scale_))\nprint('observation_noise_variance: {}'.format(observation_noise_variance_))")


# In[36]:


# Plot the loss evolution
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()


# In[39]:


# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
predictive_index_points_1 = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_1 = predictive_index_points_1[..., np.newaxis]

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,  # Reuse the same kernel instance, with the same params
    index_points=predictive_index_points_1,
    observation_index_points=observation_index_points_1,
    observations=observations_1,
    observation_noise_variance=observation_noise_variance,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 50
samples = gprm.sample(num_samples)


# In[43]:


# Draw samples and visualize.
samples_ = sess.run(samples)

# Plot the true function, observations, and posterior samples.
plt.figure(figsize=(12, 4))
plt.plot(predictive_index_points_1, sinusoid(predictive_index_points_1)+2,
         label='True fn')
plt.scatter(
    observation_index_points_1[:, 0], observations_1,
    label='Observations')
for i in range(num_samples):
    plt.plot(predictive_index_points_1, samples_[i, :], c='r', alpha=.1,
        label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.show()


# *Note: if you run the above code several times, sometimes it looks great and
# other times it looks terrible! The maximum likelihood training of the parameters
# is quite sensitive and sometimes converges to poor models. The best approach
# is to use MCMC to marginalize the model hyperparameters.*
