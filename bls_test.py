from ELCA import transit, lc_fitter

from bls import BLS

import matplotlib.pyplot as plt
import numpy as np

class fitData():
    def __init__(self):
        self.m=[]
        self.b=[]
        self.std=[]
        self.t=[]

    def save(self,t,b,m,std):
        self.t.append(t)
        self.b.append(b)
        self.m.append(m)
        self.std.append(std)
    def get(self,i):
        return self.t[i],self.b[i],self.m[i],self.std[i]

def phase_bin(time,flux,per,tmid=0,cadence=16,offset=0.25):
    '''
        Phase fold data and bin according to time cadence
        time - [days]
        flux - arbitrary unit
        per - period in [days]
        tmid - value in days
        cadence - spacing to bin data to [minutes] 
    '''
    phase = ((time-tmid)/per + offset)%1

    sortidx = np.argsort(phase)
    sortflux = flux[sortidx]
    sortphase = phase[sortidx]

    cad = cadence/60./24/per # phase cadence according to kepler cadence
    pbins = np.arange(0,1+cad,cad) # phase bins
    bindata = np.zeros(pbins.shape[0]-1)
    for i in range(pbins.shape[0]-1):
        pidx = (sortphase > pbins[i]) & (sortphase < pbins[i+1])

        if pidx.sum() == 0 or np.isnan(sortflux[pidx]).all():
            bindata[i] = np.nan
            continue

        bindata[i] = np.nanmean(sortflux[pidx])

    phases = pbins[:-1]+np.diff(pbins)*0.5

    # remove nans
    #nonans = ~np.isnan(bindata)
    #return phases[nonans],bindata[nonans]
    return phases, bindata

def BLLS(t,f, fmodel, periods, q_range):

    # alloc data
    pdata = fitData()
    pdata.p = []; pdata.q = []
    for p in periods:

        # compute phase and sort data
        phase,data = phase_bin(t,f,p,min(t),cadence=2,offset=0.25)
        sidx = np.argsort(phase)
        sphase = phase[sidx]
        sdata = data[sidx]
        nonans = ~np.isnan(sdata)

        # loop through fractions of phase
        qdata = fitData()
        qdata.q = []
        for q in q_range:

            # construct model
            model = np.zeros(sdata.shape)
            model[sphase<q] = 1 # simple box
            model[model==1] = fmodel(sphase[model==1]/max(sphase[model==1])) # create a transit-like shape

            # marginalize over t0 
            tdata = fitData()
            for i in np.arange(0,len(sdata), int( (model>0).sum() * 0.2) ):
                smodel = np.roll(model,i) # replace with shift 
                
                # pull out sub region of data around model to optimize linalg?
                # solve linear least squares to optimize transit pars
                A = np.vstack([np.ones(len(smodel[nonans])), smodel[nonans]]).T
                b, m = np.linalg.lstsq(A, sdata[nonans], rcond=None)[0]
                y = b + m*smodel[nonans]
                tdata.save(i, b, m, np.std(sdata[nonans]-y))

            # figure out why the amplitude is not close to my expected SNR
            # pick the best fit from t0
            snr = np.array(tdata.m)*-1 / np.array(tdata.std)
            qdata.save( *tdata.get(np.argmax(snr)) )
            qdata.q.append(q)
        
        # pick the best fit from transit duration
        snr = np.array(qdata.m)*-1 / np.array(qdata.std)
        pdata.save( *qdata.get(np.argmax(snr)) )
        pdata.q.append(q)
        pdata.p.append(p)
        
    # compute final snr for BLLS
    return pdata

if __name__ == "__main__":

    t = np.linspace(0,20, int(20*24*60*0.5) )
    NOISE = 7.5e-4

    # randomly delete data 
    t = np.random.choice(t, int(0.9*t.shape[0]) )
    t = np.sort(t)

    init = { 'rp':np.random.normal(0.05,0.01),  # Rp/Rs
            'ar':np.random.normal(15,0.1),      # a/Rs
             'per':np.random.normal(3.5,0.1),   # period [days]
             'inc':89.5,                # Inclination [deg]
             'u1': 0.3, 'u2': 0.1,      # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,          # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,            # Airmass extinction terms
             'a2':0, 'tm':3.5*0.25 }    # tm = Mid Transit time (Days)

    # only report params with bounds, all others will be fixed to initial value
    mybounds = {
              'rp':[0,1],
              'tm':[min(t),max(t)],
              'a0':[-np.inf,np.inf],
              'a1':[-np.inf,np.inf]
    }

    # GENERATE NOISY DATA
    tmodel = transit(time=t, values=init)  
    data = tmodel + np.random.normal(0, NOISE, len(t))
    dataerr = np.random.normal(400e-6, 50e-6, len(t))

    # scale a transit model between 0 and 1
    stmodel = ((tmodel-tmodel.min() )/ max(tmodel-tmodel.min()))[:2000]
    intrans = -1*(stmodel[stmodel<1]-1)
    xp = np.linspace(0,1, len(intrans) )
    fmodel = lambda x : np.interp(x, xp, intrans)

    # search for periodic signals
    # compute the phase space to search 
    pdata = BLLS(t,data, fmodel, np.linspace(2,10,250), np.linspace(0.05,0.15,50))
    snr = np.array(pdata.m)*-1 / np.array(pdata.std)

    # figure out BLS scaling to SNR 
    bls = BLS(t, data, np.ones(dataerr.shape[0]), period_range=(2,10), q_range=(0.05, 0.15), nf=500, nbin=100)
    res = bls()
    periods = 1./bls.freqs

    fig = plt.figure()
    ax0 = plt.subplot2grid((1,3),(0,0), colspan=2)
    ax1 = plt.subplot2grid((1,3),(0,2))
    #ax2 = plt.subplot2grid((2,2),(1,1))

    ax0.plot(t,data,'k.',alpha=0.5)
    ax0.set_title("Test Data")
    ax0.set_ylabel("Relative Flux")
    ax0.set_xlabel("Time [day]")
    ax1.plot(periods,res.sde,'k-',label='BLS (Kovacs 2002)')
    ax1.set_xlabel('Period [day]')
    #ax1.set_ylabel('SNE')
    ax1.set_title("Transit Periodogram")
    ax1.plot(pdata.p,snr,label='Custom')
    ax1.set_xlabel("Period [day]")
    ax1.set_ylabel("S/N")
    ax1.plot([min(pdata.p),max(pdata.p)],[init['rp']**2/NOISE,init['rp']**2/NOISE],'k--',label='Truth')
    ax1.legend(loc='best')
    ax1.set_ylim([0,5])
    plt.tight_layout()
    plt.show()
    '''
    #dataerr = np.random.normal(400e-6, 50e-6, len(t))
    myfit = lc_fitter(t,data,
                        dataerr=dataerr,
                        init= init,
                        bounds= mybounds,
                        nested=True,
                        loss='soft_l1'
                        )
    for k in myfit.data['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['NS']['parameters'][k],myfit.data['NS']['errors'][k]) )

    myfit.plot_results(show=True,t='NS')
    myfit.plot_posteriors(show=True)
    '''