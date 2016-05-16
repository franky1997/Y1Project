import numpy as np
import matplotlib.pyplot as plt

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
nbkg = 1000
nsig = 1000

# Mass range and bin properties
min = 100.
max = 180.

# Produce pseudodata
# Note this is a wonky model that's not the same as the fit model later
# It's more like real data that way!
np.random.seed(seedval)
signal = np.random.normal(mh,res,nsig)

powerlaw = lambda x, amp, index: amp * (x**index)


xs = np.linspace(1.1, 180.1, nbkg) 
bkg = 8*powerlaw(xs, 180.0, -0.5)+190    # simulated perfect data
yerr = 0.0002 * bkg                      # simulated errors (10%)
bkg += np.random.randn(nbkg) * yerr       # simulated noisy data

all = np.concatenate((signal,bkg))
    

# Build pseudodata histogram and plot axes
min = 100.
max = 180.
binwidth = 1.
bincenters = bincenters = np.linspace(min+0.5,max-0.5,(max-min)/binwidth)
binedges=np.linspace(min,max,(max-min)/binwidth+1)
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

# Plot pseudodata histogram 

plt.xlim(min,max)


plt.show()
