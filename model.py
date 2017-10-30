import numpy as np
import inspect
import minuit2 as minuit
from scipy import stats,optimize

class Model:
    def __init__(self,weighter,sim_data,bins=None,range=None,aux_data=None):
        self.weighter = weighter
        self.sim_data = sim_data
        self.aux_data = aux_data
        self.ndim = self.sim_data.shape[1]

        if bins is None:
            bins = [1]*sim_data.shape[1]

        self.bin_edges = np.histogramdd(self.sim_data,bins=bins,range=range)[1]
        self.bins = [len(b)-1 for b in self.bin_edges]

        self.multi_indices = [np.digitize(self.sim_data[:,col],self.bin_edges[col])-1 for col in xrange(self.ndim)]

        cut = np.prod([(self.multi_indices[col]<self.bins[col])&(self.multi_indices[col]>-1) for col in xrange(self.ndim)],axis=0).astype(bool)

        self.sim_data = self.sim_data[cut]
        self.aux_data = self.aux_data[cut]
        self.multi_indices = [m[cut] for m in self.multi_indices]

        self.indices = np.ravel_multi_index(self.multi_indices,self.bins)

        cut = np.argsort(self.indices)

        self.sim_data = self.sim_data[cut]
        self.aux_data = self.aux_data[cut]
        self.multi_indices = [m[cut] for m in self.multi_indices]
        self.indices = self.indices[cut]

        self.weighter = weighter
    
    def weights(self,**values):        
        return self.weighter(self.aux_data,**values)

    def expectations(self,**values):        
        return np.bincount(self.indices,weights=self.weights(**values))
    
    def multi_expectations(self,**values):
        return np.reshape(self.expectations(**values),self.bins)

class Likelihood:
    def __init__(self,data,model,prior=None,llh='poisson'):
        self.llh = llh
        self.prior = prior
        self.model = model
        self.data = data
        self.multi_counts = np.histogramdd(self.data,bins=self.model.bin_edges)[0]
        self.counts = self.multi_counts.flatten()

        params = inspect.getargspec(self.model.weighter).args
        params.remove('aux_data')

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

        self.minuit = minuit.Minuit2(fcn)

        self.seeds = dict(zip(params,inspect.getargspec(self.model.weighter).defaults))
        self.minuit.values = self.seeds.copy()

    def __call__(self,**values):
        if self.llh == 'poisson':
            mu = self.model.expectations(**values)
            llh = 2.*np.sum(np.where(mu<=0,0.,
                                     np.where(self.counts<=0,mu,
                                              mu - self.counts + self.counts*(np.log(self.counts) - np.log(mu)))))
        elif self.llh == 'gaussian':
            mu = self.model.expected_counts(**values)
            llh = np.sum(np.where(mu<=0,0.,(mu - self.counts)**2/mu))

        elif self.llh == 'dima':
            weights = self.model.weights(**values)
            
            xi = np.zeros(np.prod(self.model.bins))
            xi_w = xi[self.model.indices]
            f = self.counts*np.bincount(self.model.indices,weights=weights/(1.+xi_w*weights)) - (1 - xi)
            dfdxi = self.counts*np.bincount(self.model.indices,weights=weights**2/(1.+xi_w*weights)**2)/np.bincount(self.model.indices,weights=weights/(1.+xi_w*weights)) + 1.
            print f
            print dfdxi
            xi_new = xi - f/dfdxi
            while np.amax(abs((xi_new-xi)/xi_new)) > 0.01:
                print xi
                xi = xi_new
                xi_w = xi[self.model.indices]
                f = self.counts*np.bincount(self.model.indices,weights=weights/(1.+xi_2*weights)) - (1 - xi)
                dfdxi = self.counts*np.bincount(self.model.indices,weights=weights**2/(1.+xi_w*weights)**2)/np.bincount(self.model.indices,weights=weights/(1.+xi_w*weights)) + 1.
                xi_new = xi - f/dfdxi

            xi = xi_new
            xi_w = xi[self.model.indices]

            llh = 2.*(np.sum(np.log(1+xi_w*weights)) + np.sum(self.counts*np.log(1-xi)))
        else:
            raise RuntimeError('Likelihood model not recognized!')

        return llh

    def profile(self,**values):
        fixed0 = self.minuit.fixed.copy()
        for key in values.iterkeys():
            self.minuit.fixed[key] = True
        self.minuit.values.update(values)
        self.minuit.migrad()
        self.minuit.fixed = fixed0.copy()
        return self.minuit.fval

    def scan(self,**values):
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
