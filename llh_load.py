import numpy as np
from scipy import linalg
from model import Model,Likelihood

nu = np.load('NuGen_HESE_l2.npy')

reco = 'Monopod'

casc_cut = (nu['Tag']==0.)&(nu['log10CausalQTot']>np.log10(1500))&(nu['deltaLLH']<0)&(((nu['log10'+reco+'Energy'] > 4.333333)&(nu['cos'+reco+'Zenith']>0))|(nu['cos'+reco+'Zenith']<0))
track_cut = (nu['Tag']==0.)&(nu['log10CausalQTot']>np.log10(6000))&(nu['deltaLLH']>0)&(nu['log10MillipedeEnergy']>4.333333)
cut = casc_cut|track_cut

nu_data = np.array(nu[['log10'+reco+'Energy','cos'+reco+'Zenith','deltaLLH']][cut].tolist())
nu_data[track_cut[cut]]= np.array(nu[['log10MillipedeEnergy','cosMCZenith','deltaLLH']][track_cut].tolist())

nu_aux_data = np.array(nu[['log10TrueEnergy','AstrophysicalWeight','HondaGaisserH3aWeight','Analytic1TeVConventionalGaisserH3aSelfVetoProb','EnbergGaisserH3aWeight','Analytic1TeVPromptGaisserH3aSelfVetoProb','Type']][cut].tolist())

muon =  np.load('MuonGun_HESE_l2.npy')

casc_cut = (muon['Tag']==0.)&(muon['log10CausalQTot']>np.log10(1500))&(muon['deltaLLH']<0)&(((muon['log10'+reco+'Energy'] > 4.333333)&(muon['cos'+reco+'Zenith']>0))|(muon['cos'+reco+'Zenith']<0))&(muon['cos'+reco+'Zenith']>0)
track_cut = (muon['Tag']==0.)&(muon['log10CausalQTot']>np.log10(6000))&(muon['deltaLLH']>0)&(muon['log10MillipedeEnergy']>4.333333)
cut = casc_cut|track_cut 


muon_data = np.array(muon[['log10'+reco+'Energy','cos'+reco+'Zenith','deltaLLH']][cut].tolist())
muon_data[track_cut[cut]]= np.array(muon[['log10MillipedeEnergy','cosMuonZenith','deltaLLH']][track_cut].tolist())

muon_aux_data = np.array(muon[['MuonWeight','deltaLLH']][cut].tolist())
muon_aux_data[:,0][casc_cut[cut]] = muon_aux_data[:,0][casc_cut[cut]]/np.sum(muon['MuonWeight'][(muon['Tag']==0.)&(muon['log10CausalQTot']>np.log10(1500))&(muon['deltaLLH']<0)&(((muon['log10'+reco+'Energy'] > 4.333333)&(muon['cos'+reco+'Zenith']>0))|(muon['cos'+reco+'Zenith']<0))])
muon_aux_data[:,0][track_cut[cut]] = muon_aux_data[:,0][track_cut[cut]]/np.sum(muon['MuonWeight'][track_cut])

sim_data = np.vstack((nu_data,muon_data))
aux_data = linalg.block_diag(nu_aux_data,muon_aux_data)

tag_sca = 2.4

def weighter(aux_data,
             astro_norm=1.,
             f_etau=0.5,
             f_mu=0.333333,
             astro_index=2.,
             astro_cutoff=100,
             conv_norm=1.,
             prompt_norm=1.,
             muon_casc_norm=tag_sca*4,
             muon_track_norm=tag_sca*4):

    energies = 10**aux_data[:,0]
    astro_weights = aux_data[:,1]
    conv_weights = aux_data[:,2]*aux_data[:,3]
    prompt_weights = aux_data[:,4]*aux_data[:,5]
    ptypes = aux_data[:,6]
    muon_weights = aux_data[:,7]
    delta_llh = aux_data[:,8]
    nue = np.in1d(ptypes,[66,67])
    numu = np.in1d(ptypes,[68,69])
    nutau = np.in1d(ptypes,[133,134])
    f_e = f_etau*(1-f_mu)
    f_tau = 1 - f_e - f_mu
    return 3.*astro_norm*(f_e*nue+f_mu*numu+f_tau*nutau)*astro_weights*(energies/1e5)**(2.-astro_index)*np.exp(-energies/(astro_cutoff*1e6)) + conv_norm*conv_weights+prompt_norm*prompt_weights + (muon_casc_norm*(delta_llh<0) + muon_track_norm*(delta_llh>0))*muon_weights

bins = [15,4,2]
range = [[3.666666,7],[-1.000001,1.000001],[-1e6,1e6]]
model = Model(weighter,sim_data,bins=bins,range=range,aux_data=aux_data)

dat = np.load('Data_l2.npy')

casc_cut = (dat['Tag']==0.)&(dat['log10CausalQTot']>np.log10(1500))&(dat['deltaLLH']<0)&(((dat['log10'+reco+'Energy'] > 4.333333)&(dat['cos'+reco+'Zenith']>0))|(dat['cos'+reco+'Zenith']<0))

casc_data = np.array(dat[['log10'+reco+'Energy','cos'+reco+'Zenith','deltaLLH']][casc_cut].tolist())

track_dat = np.load('Data_HESE.npy')
track_data = np.array(track_dat[['log10MillipedeEnergy','cosMillipedeZenith','deltaLLH']][track_dat['deltaLLH']>0].tolist())

data = np.vstack((casc_data,track_data))

def prior(**values):
    return (values['muon_casc_norm'] - tag_sca*4)**2/(tag_sca**2*4) + (values['muon_track_norm'] - tag_sca*4)**2/(tag_sca**2*4)

llh = Likelihood(data,model,prior=prior,llh='dima')

llh.minuit.limits = {'astro_norm':(0,100),
                     'f_etau':(0,1),
                     'f_mu':(0,1),
                     'astro_index':(1,4),
                     'astro_cutoff':(0.1,100),
                     'conv_norm':(0,100),
                     'prompt_norm':(0,100),
                     'muon_casc_norm':(0,100),
                     'muon_track_norm':(0,100)}

# llh.minuit.fixed['astro_index'] = True
llh.minuit.fixed['astro_cutoff'] = True
llh.minuit.fixed['f_etau'] = True
llh.minuit.fixed['f_mu'] = True
