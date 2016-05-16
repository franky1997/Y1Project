
from pylab import *
from scipy import *

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)


##########
# Generate data points with noise
##########
num_points = 500000

# Note: all positive, non-zero data
xdata = np.linspace(1.1, 180.1, num_points) 
ydata = 8*powerlaw(xdata, 180.0, -0.5)+190    # simulated perfect data
yerr = 0.0002 * ydata                      # simulated errors (10%)

ydata += randn(num_points) * yerr       # simulated noisy data


errorbar(xdata, ydata, yerr=yerr,fmt='k.')  # Data


title('Best Fit Power Law')
xlabel('X')
ylabel('Y')
xlim(0, 180)
plt.show()