import numpy as np
from inspect import getargspec
from scipy import optimize, stats
import minuit2

class ModelHistogram(object):
    """
    Computes histograms of weighted Monte Carlo data.
    """
    def __init__(self, data, columns, weights=None, bins=None, range=None):
        """
        :param data: numpy recarray whose rows contain event level information
        :param columns: the column names to be histogrammed
        ;param weights: a function computing event weights from data
        :param bins: list of bin edge arrays or tuple of bin numbers
        :param range: list of lower and upper limits for each axis        
        """
        self.columns = columns
        self.bin_edges = np.histogramdd(np.array(data[columns].tolist()), bins=bins, range=range)[1]
        self.ndim = len(self.bin_edges)
        self.bins = [len(b)-1 for b in self.bin_edges]
        self.range = [[b.min(),b.max()] for b in self.bin_edges]
        for column,r in zip(self.columns,self.range):
            data = data[(data[column] >= r[0])&(data[column] <= r[1])]
        self.bin_multi_index = [np.digitize(data[column]) - 1 for axis in self.columns]
        self.bin_index = np.ravel_multi_index(multi_index, self.bins)
        self.index = np.arange(len(data))
        self.data = data

        self.weights = lambda **values: weights(self.data,**values)

        self.params = getargspec(self.weights).args
        self.seeds = dict(zip(self.params,getargspec(self.weights).defaults))

    def expectations(self, **values):
        """
        Compute the expected number of counts per bin.
        """
        return np.bincount(self.bin_index, weights=self.weights(**values), minlength=np.prod(self.bins)).reshape(self.bins)

    def uncertainties(self, **values):
        """
        Compute the uncertainty on the expected number of counts per bin.
        """
        return np.sqrt(np.bincount(self.bin_index, weights=self.weights(**values)**2, minlength=np.prod(self.bins)).reshape(self.bins))        
        
class LikelihoodFit(object):
    """
    Compute likelihood from a MC histogram and experimental data.
    """
    def __init__(self, data, model, prior=None, llh='poisson'):
        """
        :param data: numpy recarray containing data to be fit
        :param model: ModelHistogram used to fit data
        :param prior: function describing a prior knowledge on parameters to be added to the likelihood
        :param llh: type of liklihood, 'poisson', 'gaussian', or 'dima'
        """
        if llh!='poisson':
            raise NotImplementedError("Only Poisson likelihood supported so far")

        self.llh = llh
        self.prior = prior
        self.model = model
        self.data = data

        self.counts = np.histogramdd(np.array(self.data[self.model.columns].tolist()),
                                     bins=self.model.bin_edges)[0]
        
        fcn_str = "fcn = lambda "
        for param in params:
            fcn_str += param+","
        fcn_str = fcn_str[:-1] + ": self.__call__(**{"
        for param in params:
            fcn_str += "'"+param+"':"+param+","
        fcn_str = fcn_str[:-1] + "})"
        if not prior is None:
            fcn_str += " + self.prior(**{"
            for param in params:
                fcn_str += "'"+param+"':"+param+","
            fcn_str = fcn_str[:-1] + "})"        

        exec fcn_str in locals()

        self.minuit = minuit2.Minuit2(fcn)
        self.minuit.values = self.model.seeds

    def __call__(self,**values):
        """
        Calculate likelihood
        """
        mu = self.model.expectations(**values)
        llh = 2.*np.sum(np.where(mu<=0,0.,
                                 np.where(self.counts<=0,mu,
                                          mu - self.counts + self.counts*(np.log(self.counts) - np.log(mu)))))

    def profile(self,**values):
        """
        Calculate profile likelihood
        """
        fixed0 = self.minuit.fixed.copy()
        for key in values.iterkeys():
            self.minuit.fixed[key] = True
        self.minuit.values.update(values)
        self.minuit.migrad()
        self.minuit.fixed = fixed0.copy()
        return self.minuit.fval

    def scan(self,**values):
        """
        Scan the profile liklihood over a grid of values
        """
        values0 = self.minuit.values.copy()
        fixed0 = self.minuit.fixed.copy()
        for key in values.iterkeys():
            self.minuit.fixed[key] = True
        values = dict(zip(values.keys(),np.meshgrid(*values.values())))
        llh = np.shape(values.values()[0])
        for index in np.ndindex(llh.shape):
            llh[index] = self.profile(**{key:values[key][index] for key in values.iterkeys()})
        self.minuit.values = values0.copy()
        self.minuit.fixed = fixed0.copy()
        return llh


    def profile_CI_1d(self,param,CL):
        """
        Calculate approximate Wilks' confidence intervals

        :param param: parameter name
        :param CL: confidence level
        """
        if not param in self.minuit.limits.keys():
            raise RuntimeError('You must specify parameter limits.')

        values0 = self.minuit.values.copy()
        fixed0 = self.minuit.fixed.copy()
        delta_llh = stats.chi2.ppf(CL,1)
        self.minuit.migrad()
        best_llh  = self.minuit.fval
        best_param = self.minuit.values[param]

        if self.profile(**{param:self.minuit.limits[param][0]}) - best_llh > delta_llh:
            lo_lim = optimize.brentq(lambda x: self.profile(**{param:x}) - best_llh - delta_llh,self.minuit.limits[param][0],best_param)
        else:
            lo_lim = self.minuit.limits[param][0]
        if self.profile(**{param:self.minuit.limits[param][1]}) - best_llh > delta_llh:
            up_lim = optimize.brentq(lambda x: self.profile(**{param:x}) - best_llh - delta_llh,best_param,self.minuit.limits[param][1])
        else:
            up_lim = self.minuit.limits[param][1]

        self.minuit.values = values0.copy()
        self.minuit.fixed = fixed0.copy()
        return np.array([lo_lim,up_lim])
