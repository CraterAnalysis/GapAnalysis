"""
Created on XXXXX

@author: Stuart J. Robbins

XXXXX

"""

##TO DO LIST:


#Import the libraries needed for this program.
import argparse                     #allows parsing of input options
import numpy as np                  #lots of maths
import scipy as sp                  #lots of special maths
import scipy.stats as spStats       #special maths!
import math                         #other lots of maths
import time                         #timing for debugging
import sys                          #for setting NumPY output for debugging
import matplotlib.pyplot as mpl     #for display purposes
from matplotlib.colors import LinearSegmentedColormap #custom colorscale

#Set any code-wide parameters.
np.set_printoptions(threshold=sys.maxsize) #for debugging purposes


#General help information.
parser = argparse.ArgumentParser(description='Identify polygonal crater rims.')

#Various runtime options.
parser.add_argument('--input',      dest='inputFile',   action='store', default='',         help='Name of the CSV file with a single crater diameter per line (assumes units of kilometers).')
parser.add_argument('--diam_min',   dest='d_DIAM_min',  action='store', default='0.03125',  help='The minimum crater diameter that will be analyzed (if smaller than the diameters supplied, will be over-ridden to that value).')
parser.add_argument('--diam_max',   dest='d_DIAM_max',  action='store', default='1000',     help='The maximum crater diameter that will be analyzed (if larger than the diameters supplied, math will still assume THIS cutoff diameter).')

#Parse all the arguments so they can be stored
args=parser.parse_args()

#Store the time to output it at the end.
timer_start = time.time()
tt = time.time()




##----------------------------------------------------------------------------##

def find_nearest(array,value):
    array = [abs(array[i]-value) for i in range(0,len(array))]
    idx = array.index(min(array))
    return idx



##------------------------- USER-MODIFIED VARIABLES --------------------------##

d_DIAM_min                  = float(args.d_DIAM_min)   #what is the minimum CEIL(diameter) bin - note that this will be smaller in the end due to bin centering
d_DIAM_max                  = float(args.d_DIAM_max)    #number of runs to do for confidence bands
i_MonteCarlo_conf_interval  = 1000       #number of runs to do for confidence bands before checking for convergence
d_confidence                = 2         #sigmas of confidence band
f_output_time_diagnostics   = 0         #0 = no, 1 = yes
f_MT                        = 0         #0 = no to multi-thread the bootstrap; 1 = yes to do the multi-thread


##-------- SET UP THE DATA AND IMPORTANT VARIABLES AND VECTORS/ARRAYS --------##


#Read the crater data into the array DIAM_TEMP and DIAM_TEMP_SD.  The read-in
# via numpy removes any NaN values (missing rows), so we do not need to check..
DIAM_TEMP = np.genfromtxt(args.inputFile,delimiter=',')

#Locate the minimum and maximum diameters.
d_diam_min_loc = int(math.ceil(np.searchsorted(DIAM_TEMP,d_DIAM_min)))
d_diam_max_loc = int(math.ceil(np.searchsorted(DIAM_TEMP,d_DIAM_max)))

#Sort the diameters (will save time later).
DIAM_TEMP.sort()

#Now that we're done setting up the craters, set other important variables.
# The min and max here will be the min/max diameters to which the KDE
# KDE calculation is carried out.  The total craters saves a tiny amount of time
# so the code doesn't have to keep calculating it.
i_total_craters = len(DIAM_TEMP)



##--------- FIT AN UNTRUNCATED PARETO DISTRIBUTION (D_MIN, NO D_MAX) ---------##

#Maximum Likelihood Frequentist Approach for a regular, untruncated Pareto.
summation = 0   #needs to be calculated for untruncated and truncated
for iCounter in range(d_diam_min_loc,len(DIAM_TEMP)):
    summation += math.log(DIAM_TEMP[iCounter]/d_DIAM_min)    #this is the inner sum for any maximum likelihood estimate of the Pareto or truncated Pareto
pareto_slope_DSFD = -((len(DIAM_TEMP)-d_diam_min_loc)/summation + 1)
pareto_slope_CSFD = pareto_slope_DSFD + 1
pareto_slope_uncertainty = (len(DIAM_TEMP)-d_diam_min_loc)/summation / math.sqrt(len(DIAM_TEMP)-d_diam_min_loc)
print("Average slope from ML-frequentist to an untruncated fit:   %g±%g\r" % (pareto_slope_DSFD, pareto_slope_uncertainty))



##------------ FIT A TRUNCATED PARETO DISTRIBUTION (D_MIN, D_MAX) ------------##

#Calculate fit parameter for a truncated Pareto distribution.  This uses eq. 4
# from Aban et al. (2006), but the equation is a condition set equal to 0 and it
# cannot be solved analytically, it must be solved numerically.
resolution  = 1e-4  #for the truncated Pareto, to what accuracy do you want to solve for alpha
min_guess   = resolution    #for the truncated Pareto, what is the minimum guess (division by 0 issues if this is negative)
max_guess   = +10   #for the truncated Pareto, what is the maximum guess (remember that alpha is actually -(alpha-1) so for a slope of -3, alpha will be +2)
#Variable a, b                                //priors for the Bayesian approach to MLE fitting
#Variable finished    = 100                    //for the Bayesian approach, how many iterations, max
#Variable summation = 0                    //dummy variable for Maximum Likelihood
#Variable display_confidence = 1        //sigma for display confidence, if displaying
guesses = [0]*(int((max_guess-min_guess)/resolution+1)+1)
guesses = [min_guess + x*resolution for x in range(0,len(guesses)-1)]
guesses[:] = [(d_diam_max_loc-d_diam_min_loc) / guesses[x] + (d_diam_max_loc-d_diam_min_loc)*np.power((d_DIAM_min/d_DIAM_max),guesses[x])*np.log(d_DIAM_min/d_DIAM_max)/(1-np.power((d_DIAM_min/d_DIAM_max),guesses[x])) - summation for x in range(0,len(guesses)-1)]

#Find the value closest to 0 -- we minimize the likelihood function.
zero_point = find_nearest(guesses,0)
#Determine the fit values.
if (zero_point > 0) and (zero_point < len(guesses)-1):
    alpha           = min_guess + zero_point*resolution
    fit_exponent    = -(alpha+1)
    fit_uncertainty = np.power((d_DIAM_min/d_DIAM_max),alpha) * np.power(math.log(d_DIAM_min/d_DIAM_max),2) / np.power(1-np.power((d_DIAM_min/d_DIAM_max),alpha),2)
    fit_uncertainty = 1 / (1/(alpha*alpha) - fit_uncertainty)
    fit_uncertainty = math.sqrt(fit_uncertainty / (d_diam_max_loc-d_diam_min_loc))
    print("Average slope from ML-frequentist to a    truncated fit:   %g±%g\r" % (fit_exponent, fit_uncertainty))
else:
    print("Need to expand guesses for truncated Pareto distribution, zero was not found.")

#Release RAM.
guesses = []



##------------------- CREATE A RANDOM PARETO DISTRIBUTION --------------------##

#Set the distribution's values.
power = fit_exponent
power = -(power+1)

#Create the random sample, multiple times.
randomsample = [[d_DIAM_min / np.power(np.random.rand(),1/power) for i in range(d_diam_min_loc,d_diam_max_loc)] for j in range(0,i_MonteCarlo_conf_interval)]



##------------------ CALCULATE SAMPLE COMPARISON STATISTICS ------------------##

#Calculate Anderson-Darling 2-sample statistic, and Kolmagorov-Smirnov 2-sample
# statistic p-values.
testResults = [spStats.anderson_ksamp([DIAM_TEMP[d_diam_min_loc:d_diam_max_loc], randomsample[:][i]]) for i in range(0,i_MonteCarlo_conf_interval)]
ADmean = 0
for iCounter in range(0,i_MonteCarlo_conf_interval): ADmean += testResults[iCounter][2]
print("2-Sample Anderson-Darling test   p-value:", ADmean/i_MonteCarlo_conf_interval)
testResults = [spStats.ks_2samp(DIAM_TEMP[d_diam_min_loc:d_diam_max_loc], randomsample[:][i]) for i in range(0,i_MonteCarlo_conf_interval)]
KSmean = 0
for iCounter in range(0,i_MonteCarlo_conf_interval): KSmean += testResults[iCounter][1]
print("2-Sample Kolmogorov–Smirnov test p-value:", KSmean/i_MonteCarlo_conf_interval)


randomsample_sorted = []
randomsample_sorted_median = []
randomsample_sorted_02 = []
randomsample_sorted_97 = []
percentile_Mimas_D = []
for iCounter in range(0,i_MonteCarlo_conf_interval):
    dummy = randomsample[iCounter][:]
    dummy.sort()
    randomsample_sorted.append(dummy)
for iCounter in range(0,d_diam_max_loc-d_diam_min_loc):
    dummy = []
    for jCounter in range(0,i_MonteCarlo_conf_interval): dummy.append(randomsample_sorted[jCounter][iCounter])
    dummy.sort()
    randomsample_sorted_02.append(dummy[int(len(dummy)*0.025)])
    randomsample_sorted_97.append(dummy[int(len(dummy)*0.975)])
    randomsample_sorted_median.append(np.median(dummy))
    percentile_Mimas_D.append(abs(0.5-np.searchsorted(dummy, DIAM_TEMP[d_diam_min_loc+iCounter])/i_MonteCarlo_conf_interval)*2.)
#    print(randomsample_sorted_02[len(randomsample_sorted_02)-1],randomsample_sorted_median[len(randomsample_sorted_median)-1],randomsample_sorted_97[len(randomsample_sorted_97)-1])



##-------- CALCULATE ANOTHER TEST, USING CUMULATIVE DENSITY FUNCTION ---------##

#Create one large random sample.
randomsample_aggregate = np.resize(randomsample,((d_diam_max_loc-d_diam_min_loc)*i_MonteCarlo_conf_interval))
randomsample_aggregate.sort()
probability = np.searchsorted(randomsample_aggregate,DIAM_TEMP[len(DIAM_TEMP)-1])-np.searchsorted(randomsample_aggregate,DIAM_TEMP[len(DIAM_TEMP)-2])
probability /= len(randomsample_aggregate)
probability = math.exp(-(d_diam_max_loc-d_diam_min_loc)*probability)
print("Probability of NOT observing craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))



##-------------------- OUTPUT INTERPRETATION TO THE USER ---------------------##

print("\n\n               ~~~~~ INTERPRETATION ~~~~~\n")
print(" Based on the truncated Pareto fit, and comparison to that, we estimate the\n probability that the hypothesis that the gap between the two largest craters\n is NOT meaningful can be rejected.")
if ADmean/i_MonteCarlo_conf_interval < 0.01:
    print("  - From the K-Sample Anderson-Darling test, the null hypothesis CAN be rejected.")
else:
    print("  - From the K-Sample Anderson-Darling test, the null hypothesis canNOT be rejected.")
if KSmean/i_MonteCarlo_conf_interval < 0.01:
    print("  - From the 2-Sample Kolmogorov-Smirnov test, the null hypothesis CAN be rejected.")
else:
    print("  - From the 2-Sample Kolmogorov-Smirnov test, the null hypothesis canNOT be rejected.")
if (percentile_Mimas_D[len(percentile_Mimas_D)-1] > 0.99) or (percentile_Mimas_D[len(percentile_Mimas_D)-2] > 0.99):
    print("  - From the Monte Carlo test on crater diameter range in a CSFD, the null hypothesis CAN be rejected.")
else:
    print("  - From the Monte Carlo test on crater diameter range in a CSFD, the null hypothesis canNOT be rejected.")
if probability < 0.01:
    print("  - From a basic CDF test with Poisson statistics for x=0, the null hypothesis CAN be rejected.")
else:
    print("  - From a basic CDF test with Poisson statistics for x=0, the null hypothesis canNOT be rejected.")



##--------------------------------- DISPLAY ----------------------------------##

#I'm going to comment this as though you do not know anything about Python's
# graphing capabilities with MatPlotLib.  Because I don't know anything about it

#Create the plot reference.
StackedSFD = mpl.figure(1, figsize=(10,10))

#Plot the percentile data.
for iCounter in range(0,d_diam_max_loc-d_diam_min_loc):
    if iCounter == 0:
        mpl.scatter(randomsample_sorted_median[iCounter], (d_diam_max_loc-d_diam_min_loc-iCounter), s=3, facecolors='#333333', edgecolors='#333333', label='Median Predicted', zorder=2)
        mpl.plot([randomsample_sorted_02[iCounter],randomsample_sorted_97[iCounter]], [(d_diam_max_loc-d_diam_min_loc-iCounter),(d_diam_max_loc-d_diam_min_loc-iCounter)], color='#999999', linewidth=0.5, label='95% CI', zorder=1)
    else:
        mpl.scatter(randomsample_sorted_median[iCounter], (d_diam_max_loc-d_diam_min_loc-iCounter), s=3, facecolors='#333333', edgecolors='#333333', zorder=2)
        mpl.plot([randomsample_sorted_02[iCounter],randomsample_sorted_97[iCounter]], [(d_diam_max_loc-d_diam_min_loc-iCounter),(d_diam_max_loc-d_diam_min_loc-iCounter)], color='#999999', linewidth=0.5, zorder=1)

#Plot the crater data.
pts_x = []
pts_y = []
for iCounter in range(d_diam_min_loc,d_diam_max_loc):
    pts_x.append(DIAM_TEMP[iCounter])
    pts_y.append((d_diam_max_loc-iCounter))
test = mpl.scatter(pts_x, pts_y, s=20, zorder=3, c=percentile_Mimas_D, cmap='tab20b')
test2 = mpl.colorbar(test)
test2.set_label('Crater Data; Color = probability diameter IS an outlier')
mpl.clim(0,1)

##General graph appendages.

#Append the legend to the plot.
mpl.legend(loc='upper right')

#Append axes labels.
mpl.xlabel('Crater Diameter (km)')
mpl.ylabel('Cumulative Number of Craters')

#Set scale.
mpl.ylim(0.9,np.power(10,float(math.ceil(np.log10(d_diam_max_loc-d_diam_min_loc)))))
mpl.xlim(np.power(10,float(math.floor(np.log10(min(randomsample_sorted_02))))),np.power(10,float(math.ceil(np.log10(max(randomsample_sorted_97))))))

#Make log.
mpl.loglog()

#Append graph title.
mpl.title('Significance Determination, Visualization')

#Make axes equal / square in degrees space.
#mpl.axis()


##Finally, make the plot visible.
mpl.show()
