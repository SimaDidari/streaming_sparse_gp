from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import gpflow
from gpflow import kullback_leiblers, features
from gpflow import settings
from gpflow import transforms
from gpflow.conditionals import conditional, Kuu
from gpflow.decors import params_as_tensors
from gpflow.models.model import GPModel
from gpflow.params import DataHolder
from gpflow.params import Minibatch
from gpflow.params import Parameter
from gpflow.mean_functions import Zero
float_type = settings.dtypes.float_type


class OSVGPC(GPModel):   

    def __init__(self, X, Y, kern, likelihood, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):

        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]       
        X = gpflow.params.Minibatch(X, minibatch_size, np.random.RandomState(0))
        Y = gpflow.params.Minibatch(Y, minibatch_size, np.random.RandomState(0))    
        
        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Parameter(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0] 
        self.num_data = X.shape[0] 
        self._init_variational_parameters(num_inducing=self.num_inducing, q_mu=None, q_sqrt=None, q_diag=self.q_diag)
        self.mu_old = DataHolder(mu_old)
        self.M_old = Z_old.shape[0]
        self.Su_old = DataHolder(Su_old)
        self.Kaa_old = DataHolder(Kaa_old)
        self.Z_old = DataHolder(Z_old)  

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):

        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent)) 
        #import pdb; pdb.set_trace()

        
    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)    
            
    @params_as_tensors
    def build_correction_term(self):
        # TODO
        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        Saa = self.Su_old
        ma = self.mu_old
        obj = 0
        # a is old inducing points, b is new
        mu, Sigma = self._build_predict(self.Z_old, full_cov=True)
        Sigma = Sigma[:, :, 0]
        Smm = Sigma + tf.matmul(mu, tf.transpose(mu))
        Kaa = self.Kaa_old + np.eye(Ma) * jitter
        LSa = tf.cholesky(Saa)
        LKa = tf.cholesky(Kaa)
        obj += tf.reduce_sum(tf.log(tf.diag_part(LKa)))
        obj += - tf.reduce_sum(tf.log(tf.diag_part(LSa)))

        Sainv_ma = tf.matrix_solve(Saa, ma)
        obj += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        obj += tf.reduce_sum(mu * Sainv_ma)

        Sainv_Smm = tf.matrix_solve(Saa, Smm)
        Kainv_Smm = tf.matrix_solve(Kaa, Smm)
        obj += -0.5 * tf.reduce_sum(tf.diag_part(Sainv_Smm) - tf.diag_part(Kainv_Smm))
        return obj

    
  
    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        # compute online correction termdef Kuf(feat, kern, Xnew):
        online_reg = self.build_correction_term()
        #import pdb; pdb.set_trace()

        return tf.reduce_sum(var_exp) * scale - KL + online_reg
    
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):

        mu, var = conditional(Xnew, self.Z, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, white=self.whiten)
        
        return mu + self.mean_function(Xnew), var