# S. Zenz, Imperial College
# April 2015
# A starting script to fit a simulated Higgs boson lineshape ("pseudodata")

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# If random seed set to None, random seed set at rantime so result is non-reproducible
#seedval = None

# Set value to get reproducible successful example (if parameters of script unchanged)
#seedval = 54321

# Set value to get reproducible example of failure
#seedval = 123456

# wide, but more or less ok
seedval = 123

# Properties of pseudodata
mh = 125
res = 2.
powerlawcutoff = 240

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
np.random.seed(seedval)
signal = np.random.normal(mh,res,nsig)    #(mean, scale,size)
signal = signal[signal < 180.]
bkg = (50+np.random.exponential(195,5000000))/11

#print "DEBUG", powerlawpower,nbkg
#bkg = np.random.power(powerlawpower,nbkg)
bkg = bkg[bkg < 180.]
all = np.concatenate((signal,bkg))  #join arrays

# Build pseudodata histogram and plot axes
h_all = plt.hist(all,bins=binedges,histtype='step')
plt.clf()
plt.title("Fitting to the Higgs boson")
plt.xlabel("mass")
plt.yticks([0,200,400,600,800,1000,1200,1400])
plt.ylabel("number of Higgs boson candidates")

# Plot pseudodata histogram 
# with uncertainty bands defined by the sqrt of number of entries per bin
xdata = bincenters
ydata = h_all[0]
ydataunc = np.sqrt(h_all[0])   
plt.errorbar(xdata,ydata,xerr=binwidth,yerr=ydataunc,color="b",linestyle='none',label = "Pseudodata")
plt.xlim(min,max)

#


print "True answers from pseudodata generation:"
print "\tMass:",mh
print "\tWidth:",res
print "\tNumber:",len(signal)
print

# Plot final result
xplot = bincenters


plt.legend(numpoints=1,frameon=False)
plt.show()

#(p[0])*m**-2.4)-p[1]*m**2.5
#(p[0])*m**-2)-p[1]*m**2.5
