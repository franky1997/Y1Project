# S. Zenz, Imperial College
# April 2015
# A starting script to fit a simulated Higgs boson lineshape ("pseudodata")

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy import stats
from numpy import *
from scipy import optimize
from scipy.stats import norm
from scipy.optimize import curve_fit

# If random seed set to None, random seed set at rantime so result is non-reproducible
#seedval = None

# Set value to get reproducible successful example (if parameters of script unchanged)
seedval = 54321

# Set value to get reproducible example of failure
#seedval = 123456

# wide, but more or less ok
#seedval = 123

# Properties of pseudodata
mh = 125
res = 2.
powerlawcutoff = 240.
powerlawpower = 3.5
nbkg = 50000
nsig = 1000

# Mass range and bin properties
min = 100.
max = 180.
binwidth = 1.
bincenters = bincenters = np.linspace(min+0.5,max-0.5,(max-min)/binwidth)
binedges=np.linspace(min,max,(max-min)/binwidth+1)

# Produce pseudodata
# Note this is a wonky model that's not the same as the fit model later
# It's more like real data that way!
z=[[],[]]
s=[]
for i in range(100):
    np.random.seed(i)
    signal = np.random.normal(mh,res,nsig)
    signal = signal[signal < 180.]
    bkg = (powerlawcutoff-(powerlawcutoff-min)*np.random.power(powerlawpower,nbkg))
    bkg = bkg[bkg < 180]
    
    """s = np.linspace(0, 180, 200000)  #exponential dist
    bkg = np.exp(s)-20
    berr=0.2*bkg
    bkg+=np.random.randn(200000) * berr"""
    all = np.concatenate((signal,bkg))
    
    # Build pseudodata histogram and plot axes
    h_all = plt.hist(all,bins=binedges,histtype='step')
    plt.clf()
    plt.title("Fitting to the Higgs boson")
    plt.xlabel("mass")
    plt.ylabel("number of Higgs boson candidates")
    
    # Plot pseudodata histogram 
    # with uncertainty bands defined by the sqrt of number of entries per bin
    xdata = bincenters
    ydata = h_all[0]
    ydataunc = np.sqrt(h_all[0])
    plt.errorbar(xdata,ydata,xerr=binwidth,yerr=ydataunc,color="b",linestyle='none',label = "Pseudodata")
    plt.xlim(min,max)
    
    # Define functions for fitting
    from scipy.optimize import leastsq
    def fsig(p,m):
        fitmean = p[0]
        fitsigma = p[1]
        fitnorm = p[2]
        return (fitnorm/np.sqrt(2*np.pi*fitsigma))*(np.exp(-0.5*((m-fitmean)/fitsigma)**2))
    def fbkg(p,m):
        #return (p[0]/m**2-p[1]*m**2.5) #power law
        return (p[0]+p[1]*np.exp(1/m)) #exponential
    def ftot(p,m):
        return fsig(p[:3],m)+fbkg(p[3:],m)
    def errfunc(p,m,y):
        return ftot(p,m)-y
    
    # Initial values
    # Fit is very sensitive to these - maybe you can fix this!
    p0 = [130,1.,200.,10**10,10**10]
    

    """print"""
    
    """print "\tMass:",p0[0]
    print "\tWidth:",p0[1]
    print "\tNumber:",p0[2]
    print "\tBackground params:",p0[3:]
    print
    
    print "Fitting now, you can probably ignore sqrt errors..."
    print"""
    p1, success = leastsq(errfunc,p0[:],args=(xdata,ydata))
    z[0].append(p1[2])
    z[1].append(p1[0])
    s.append(success)
"""print
print "... done fitting."
print
print "Fit results:"
print "\tMass:",p1[0]
print "\tWidth:",p1[1]
print "\tNumber:",p1[2]
print "\tBackground params:",p1[3:]"""

print "True answers from pseudodata generation:"
print "\tMass:",mh
print "\tWidth:",res
print "\tNumber:",len(signal)


# Plot final result

xplot = bincenters
plt.plot(xplot, ftot(p1, xplot), "r-",label="Fit (Total)") 
plt.plot(xplot, fbkg(p1[3:],xplot),"r.",label="Fit (Background)")
plt.legend(numpoints=1,frameon=False)
plt.clf()

def raw_num_hist():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    n, bins, patches = ax1.hist(z[0],100)
    ax1.set_xlabel("numbers")
    ax1.set_ylabel('Frequency')

def raw_mass_hist():
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    n, bins, patches = ax2.hist(z[1],100)
    ax2.set_xlabel('Mass')
    ax2.set_ylabel('Frequency')

def scatter_mass():
    fig3=plt.figure()  #scatter of success vs mass
    ax3 = fig3.add_subplot(1, 1, 1)
    xedges, yedges = np.linspace(0, 180, 1000), np.linspace(-50, 1500, 1000)
    hist, xedges, yedges = np.histogram2d(z[1], z[0], (xedges, yedges))
    xidx = np.clip(np.digitize(z[1], xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(z[0], yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]
    plt.xlabel("mass")
    plt.ylabel("success")
    plt.scatter(z[1],s,s=300,alpha=0.15, c=c)
    
def scatter_number():
    fig3=plt.figure()  #scatter of success vs mass
    ax3 = fig3.add_subplot(1, 1, 1)
    xedges, yedges = np.linspace(0, 180, 1000), np.linspace(-50, 1500, 1000)
    hist, xedges, yedges = np.histogram2d(z[1], z[0], (xedges, yedges))
    xidx = np.clip(np.digitize(z[1], xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(z[0], yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]
    plt.xlabel("number")
    plt.ylabel("success")
    plt.scatter(z[1],s,s=300,alpha=0.15, c=c)
    

def success_mass_hist():   #2d success vs mass
    fig4, ax4 = plt.subplots(figsize=(18, 18))
    plt.hist2d(z[1],s,(50, 50), cmap=plt.cm.jet)
    plt.xlabel("mass")
    plt.ylabel("success")
    plt.colorbar()
    
def success_num_hist():   #2d success vs mass
    fig4, ax4 = plt.subplots(figsize=(18, 18))
    plt.hist2d(z[0],s,(50, 50), cmap=plt.cm.jet)
    plt.xlabel("num")
    plt.ylabel("success")
    plt.colorbar()


def mass_plot():
    mass=[]
    for i in z[1]:
        if i<127.0:
            mass.append(i)
    data=mass        
    fig5, ax5 = plt.subplots(figsize=(18, 18))        
    mu, sigma = norm.fit(data)        
    plt.hist(data, bins=60, normed=True, alpha=0.6, color='r')
    xmin,xmax=plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=1)
    Mass = "Fit results:  mu = %.2f,  std = %.2f" % (mu, sigma) + "mass"
    print Mass
    plt.xlabel("Mass")
    plt.ylabel("frequency")
    plt.title(Mass)

def num_plot():
        number=[]
        for i in z[0]:
            if i>500.0:
             number.append(i)
        data=number        
        fig6, ax6 = plt.subplots(figsize=(18, 18))        
        mu, sigma = norm.fit(data)        
        plt.hist(data, bins=60, normed=True, alpha=0.6, color='g')
        xmin,xmax=plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, 'k', linewidth=1)
        s = "Fit results:  mu = %.2f,  std = %.2f" % (mu, sigma) + "number"
        print s
        print sigma
        plt.xlabel("number of Higgs")
        plt.ylabel("frequency")
        plt.title(s)
        plt.show()
        num_plot.sigma=sigma
        num_plot.number=number

        

def t():   #calculates the pull of the data set
    num_plot()
    plt.clf()
    pull=[]
    for i in num_plot.number:
        p=(1000-i)/num_plot.sigma
        pull.append(p) 
    t.pull=pull


def pull_plot():
    t()
    data=t.pull        
    fig5, ax5 = plt.subplots(figsize=(18, 18))        
    mu, sigma = norm.fit(data)        
    plt.hist(data, bins=200, normed=True, alpha=0.6, color='r')
    xmin,xmax=plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    Pull = "Fit results:  mu = %.2f,  std = %.2f" % (mu, sigma) + "pull"
    print Pull
    plt.xlabel("Pull")
    plt.ylabel("frequency")
    plt.title(Pull)


pull_plot()

plt.show()
