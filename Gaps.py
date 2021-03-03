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
parser.add_argument('--diam_max',   dest='d_DIAM_max',  action='store', default='10000',     help='The maximum crater diameter that will be analyzed (if larger than the diameters supplied, math will still assume THIS cutoff diameter).')

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
d_DIAM_max                  = float(args.d_DIAM_max)   #number of runs to do for confidence bands
i_MonteCarlo_conf_interval  = 10000       #number of runs to do for confidence bands before checking for convergence
d_confidence                = 2         #sigmas of confidence band
f_output_time_diagnostics   = 0         #0 = no, 1 = yes
f_MT                        = 0         #0 = no to multi-thread the bootstrap; 1 = yes to do the multi-thread


##-------- SET UP THE DATA AND IMPORTANT VARIABLES AND VECTORS/ARRAYS --------##


#Read the crater data into the array DIAM_TEMP and DIAM_TEMP_SD.  The read-in
# via numpy removes any NaN values (missing rows), so we do not need to check.
DIAM_TEMP = np.genfromtxt(args.inputFile,delimiter=',')

#Sort the diameters (will save time later).
DIAM_TEMP.sort()

#Locate the minimum and maximum diameters.  There are quirks to this because we
# don't want to round and risk finding things outside our range, so just do a
# linear search.
f_foundMin = False
f_foundMax = False
if DIAM_TEMP[0] > d_DIAM_min:
    f_foundMin = True
    d_diam_min_loc = 0
for iCounter in range(1,len(DIAM_TEMP)):
    if f_foundMin == False:
        if DIAM_TEMP[iCounter] > d_DIAM_min:
            f_foundMin = True
            d_diam_min_loc = iCounter       #we passed it, BUT, this is the first value in our range
    if f_foundMax == False:
        if DIAM_TEMP[iCounter] > d_DIAM_max:
            f_foundMax = True
            d_diam_max_loc = iCounter - 1   #we passed it, and this is ABOVE our range, so go down 1
if f_foundMax == False:
    d_diam_max_loc = len(DIAM_TEMP)-1       #-1 because of how Python handles indexing

i_numberfeatures = d_diam_max_loc-d_diam_min_loc +1 #since we need to include the end points
i_numberfeatures_noMax = len(DIAM_TEMP)-d_diam_min_loc #Python indexing gives us the +1 for the end
print("There are %g craters in the supplied diameter range.\r" % (i_numberfeatures))
print("There are %g craters in the diameter range but no upper bound.\r" % (i_numberfeatures_noMax))



##--------- FIT AN UNTRUNCATED PARETO DISTRIBUTION (D_MIN, NO D_MAX) ---------##

#Maximum Likelihood Frequentist Approach for a regular, untruncated Pareto.
# That means this code block uses all the features (down to the minimum), *NOT*
# the truncated set.
summation = 0   #needs to be calculated for untruncated and truncated
for iCounter in range(d_diam_min_loc,len(DIAM_TEMP)):
    summation += math.log(DIAM_TEMP[iCounter]/d_DIAM_min)    #this is the inner sum for any maximum likelihood estimate of the Pareto or truncated Pareto

pareto_slope_DSFD = -(i_numberfeatures_noMax/summation + 1)
pareto_slope_CSFD = pareto_slope_DSFD + 1
pareto_slope_uncertainty = i_numberfeatures_noMax/summation / math.sqrt(i_numberfeatures_noMax)
print("\nAverage slope from ML-frequentist to an untruncated fit (D_min – ∞        ):   %g±%g" % (pareto_slope_DSFD, pareto_slope_uncertainty))



##------------ FIT A TRUNCATED PARETO DISTRIBUTION (D_MIN, D_MAX) ------------##

#Calculate fit parameter for a truncated Pareto distribution.  This uses eq. 4
# from Aban et al. (2006), but the equation is a condition set equal to 0 and it
# cannot be solved analytically, it must be solved numerically.
resolution  = 1e-4  #for the truncated Pareto, to what accuracy do you want to solve for alpha
min_guess   = resolution    #for the truncated Pareto, what is the minimum guess (division by 0 issues if this is negative)
max_guess   = +10   #for the truncated Pareto, what is the maximum guess (remember that alpha is actually -(alpha-1) so for a slope of -3, alpha will be +2)
guesses = [0]*(int((max_guess-min_guess)/resolution+1)+1)
guesses = [min_guess + x*resolution for x in range(0,len(guesses)-1)]
guesses = [resolution if guesses[i] == 0 else guesses[i] for i in range(len(guesses))] #account for 0
summation = 0   #needs to be calculated for untruncated and truncated
for iCounter in range(d_diam_min_loc,d_diam_max_loc+1): #count up thru the maximum
    summation += math.log(DIAM_TEMP[iCounter]/d_DIAM_min)    #this is the inner sum for any maximum likelihood estimate of the Pareto or truncated Pareto
guesses[:] = [( i_numberfeatures / guesses[x] + i_numberfeatures*np.power((d_DIAM_min/d_DIAM_max),guesses[x])*np.log(d_DIAM_min/d_DIAM_max)/(1-np.power((d_DIAM_min/d_DIAM_max),guesses[x])) - summation ) for x in range(0,len(guesses))]
#print(d_diam_min_loc,d_diam_max_loc,d_diam_max_loc-d_diam_min_loc,i_numberfeatures,d_DIAM_min,d_DIAM_max,DIAM_TEMP[d_diam_min_loc],DIAM_TEMP[d_diam_max_loc],summation)

#Find the value closest to 0 -- we minimize the likelihood function.
zero_point = find_nearest(guesses,0)
#Determine the fit values.
if (zero_point > 0) and (zero_point < len(guesses)-1):
    alpha           = min_guess + zero_point*resolution
    fit_exponent    = -(alpha+1)
    fit_uncertainty = np.power((d_DIAM_min/d_DIAM_max),alpha) * np.power(math.log(d_DIAM_min/d_DIAM_max),2) / np.power(1-np.power((d_DIAM_min/d_DIAM_max),alpha),2)
    fit_uncertainty = 1 / (1/(alpha*alpha) - fit_uncertainty)
    fit_uncertainty = math.sqrt(fit_uncertainty / i_numberfeatures)
    print("Average slope from ML-frequentist to a    truncated fit (D_min – D_max    ):   %g±%g" % (fit_exponent, fit_uncertainty))
else:
    print("Need to expand guesses for truncated Pareto distribution, zero was not found.")

#Release RAM.
guesses = []


##------------------- CREATE A RANDOM PARETO DISTRIBUTION --------------------##

#Set the distribution's values.
power = fit_exponent#pareto_slope_DSFD
power = -(power+1)

#Create the random sample, multiple times.
randomsample = [[d_DIAM_min / np.power(np.random.rand(),1/power) for i in range(d_diam_min_loc,d_diam_max_loc+1)] for j in range(0,i_MonteCarlo_conf_interval)]



##------------------ CALCULATE SAMPLE COMPARISON STATISTICS ------------------##

#Calculate Anderson-Darling 2-sample statistic, and Kolmagorov-Smirnov 2-sample
# statistic p-values.  The "+1" after d_diam_max_loc is because we need to be
# INCLUSIVE of that last point.
testResults = [spStats.anderson_ksamp([DIAM_TEMP[d_diam_min_loc:d_diam_max_loc+1], randomsample[:][i]]) for i in range(0,i_MonteCarlo_conf_interval)]
ADmean = 0
for iCounter in range(0,i_MonteCarlo_conf_interval): ADmean += testResults[iCounter][2]
print("2-Sample Anderson-Darling test   p-value:", ADmean/i_MonteCarlo_conf_interval)
testResults = [spStats.ks_2samp(DIAM_TEMP[d_diam_min_loc:d_diam_max_loc+1], randomsample[:][i]) for i in range(0,i_MonteCarlo_conf_interval)]
KSmean = 0
for iCounter in range(0,i_MonteCarlo_conf_interval): KSmean += testResults[iCounter][1]
print("2-Sample Kolmogorov–Smirnov test p-value:", KSmean/i_MonteCarlo_conf_interval)



##---- CALCULATE CONFIDENCE INTERVAL ON EACH DIAMETER FOR THE NTH CRATER -----##

#Initialize arrays.
randomsample_sorted = []
randomsample_sorted_median = []
randomsample_sorted_02 = []
randomsample_sorted_97 = []
percentile_Mimas_D = []

#Use the random samples already generated, but sort them in order to calculate
# the range for a given nth diameter in the sorted list.
for iCounter in range(0,i_MonteCarlo_conf_interval):
    dummy = randomsample[iCounter][:]
    dummy.sort()
    randomsample_sorted.append(dummy)

#Determine - and store - the median and 95% range for the diameter of every nth
# crater.
for iCounter in range(0,d_diam_max_loc-d_diam_min_loc+1):
    dummy = []
    for jCounter in range(0,i_MonteCarlo_conf_interval): dummy.append(randomsample_sorted[jCounter][iCounter])
    dummy.sort()
    randomsample_sorted_02.append(dummy[int(len(dummy)*0.025)])
    randomsample_sorted_97.append(dummy[int(len(dummy)*0.975)])
    randomsample_sorted_median.append(np.median(dummy))
    percentile_Mimas_D.append(abs(0.5-np.searchsorted(dummy, DIAM_TEMP[d_diam_min_loc+iCounter])/i_MonteCarlo_conf_interval)*2.)

#Determine how many values are beyond the 95% confidence interval.
print("There are %g craters in the sample (%f%% of the sample) that are beyond the 90%% confidence interval." % (sum(i > 0.9 for i in percentile_Mimas_D), sum(i > 0.9 for i in percentile_Mimas_D)/(d_diam_max_loc-d_diam_min_loc+1)*100))
print("There are %g craters in the sample (%f%% of the sample) that are beyond the 95%% confidence interval." % (sum(i > 0.95 for i in percentile_Mimas_D), sum(i > 0.95 for i in percentile_Mimas_D)/(d_diam_max_loc-d_diam_min_loc+1)*100))
#    print(randomsample_sorted_02[len(randomsample_sorted_02)-1],randomsample_sorted_median[len(randomsample_sorted_median)-1],randomsample_sorted_97[len(randomsample_sorted_97)-1])



##----------- TEST USING COMPLIMENTARY CUMULATIVE DENSITY FUNCTION -----------##

#Create one large random sample.
probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 0
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 0 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 1
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 1 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 2
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 2 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 3
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 3 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 4
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 4 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 5
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 5 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
observed_craters = 6
probability = np.power((d_diam_max_loc+1-d_diam_min_loc)*probability,observed_craters) * math.exp(-(d_diam_max_loc+1-d_diam_min_loc)*probability) / math.factorial(observed_craters)
print("Probability of observing 6 craters in the gap (%f - %f km) is: %f" % (DIAM_TEMP[len(DIAM_TEMP)-2], DIAM_TEMP[len(DIAM_TEMP)-1], probability))



##-------------------- OUTPUT INTERPRETATION TO THE USER ---------------------##

print("\n\n               ~~~~~ INTERPRETATION ~~~~~\n")
print("Based on the truncated Pareto fit, and comparison to that, we estimate the\nprobability that the hypothesis that the gap between the two largest craters\nis NOT meaningful can be rejected as ...")
if (percentile_Mimas_D[len(percentile_Mimas_D)-1] > 0.95) or (percentile_Mimas_D[len(percentile_Mimas_D)-2] > 0.95):
    print("  - From the Monte Carlo test on crater diameter range in a CSFD, the null hypothesis CAN be rejected.")
else:
    print("  - From the Monte Carlo test on crater diameter range in a CSFD, the null hypothesis canNOT be rejected.")
print("    The largest crater was a %g percentile event, and the second-largest crater was a %g percentile event (median = 0th percentile)." % (round(percentile_Mimas_D[len(percentile_Mimas_D)-1]*100,3), round(percentile_Mimas_D[len(percentile_Mimas_D)-2]*100,3)))

probability = np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-2]),power) - np.power((d_DIAM_min/DIAM_TEMP[len(DIAM_TEMP)-1]),power)
if probability < 0.05:
    observed_craters = 0
    print("  - From a basic CDF test with Poisson statistics for x=0, the null hypothesis CAN be rejected (probability of NOT having a crater in that gap is: %g%%)." % (probability*100))
else:
    print("  - From a basic CDF test with Poisson statistics for x=0, the null hypothesis canNOT be rejected.")

print("\nA separate (but related) question is whether the sample craters were drawn\nfrom the same population as the power-law.  We can test whether that hypothesis\ncan be rejected.")
if ADmean/i_MonteCarlo_conf_interval < 0.01:
    print("  - From the K-Sample Anderson-Darling test, the null hypothesis CAN be rejected.")
else:
    print("  - From the K-Sample Anderson-Darling test, the null hypothesis canNOT be rejected.")
if KSmean/i_MonteCarlo_conf_interval < 0.01:
    print("  - From the 2-Sample Kolmogorov-Smirnov test, the null hypothesis CAN be rejected.")
else:
    print("  - From the 2-Sample Kolmogorov-Smirnov test, the null hypothesis canNOT be rejected.")



##--------------------------------- DISPLAY ----------------------------------##

#I'm going to comment this as though you do not know anything about Python's
# graphing capabilities with MatPlotLib.  Because I don't know anything about it

#Create the plot reference.
aspect =(np.log10(np.power(10,float(math.ceil(np.log10(max(randomsample_sorted_97))))))-np.log10(np.power(10,float(math.floor(np.log10(min(randomsample_sorted_02)))))))/(np.log10(np.power(10,float(math.ceil(np.log10(d_diam_max_loc+1-d_diam_min_loc)))))-np.log10(0.9))
#print(aspect)
StackedSFD = mpl.figure(1, figsize=(5,5))

#Plot the percentile data.
for iCounter in range(0,d_diam_max_loc-d_diam_min_loc+1):
    if iCounter == 0:
        mpl.scatter(randomsample_sorted_median[iCounter], (d_diam_max_loc+1-d_diam_min_loc-iCounter), s=3, facecolors='#333333', edgecolors='#333333', label='Median Predicted', zorder=2)
        mpl.plot([randomsample_sorted_02[iCounter],randomsample_sorted_97[iCounter]], [(d_diam_max_loc+1-d_diam_min_loc-iCounter),(d_diam_max_loc+1-d_diam_min_loc-iCounter)], color='#999999', linewidth=0.5, label='±95% CI', zorder=1)
    else:
        mpl.scatter(randomsample_sorted_median[iCounter], (d_diam_max_loc+1-d_diam_min_loc-iCounter), s=3, facecolors='#333333', edgecolors='#333333', zorder=2)
        mpl.plot([randomsample_sorted_02[iCounter],randomsample_sorted_97[iCounter]], [(d_diam_max_loc+1-d_diam_min_loc-iCounter),(d_diam_max_loc+1-d_diam_min_loc-iCounter)], color='#999999', linewidth=0.5, zorder=1)

#Plot the crater data.
pts_x = []
pts_y = []
for iCounter in range(d_diam_min_loc,d_diam_max_loc+1): #inclusive of the last point
    pts_x.append(DIAM_TEMP[iCounter])
    pts_y.append((d_diam_max_loc-iCounter+1))
    
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

#Make the axes equal ratio.
mpl.axis('scaled')

#Set scale.
mpl.ylim(0.9,np.power(10,float(math.ceil(np.log10(d_diam_max_loc-d_diam_min_loc)))))
mpl.xlim(np.power(10,float(math.floor(np.log10(min(randomsample_sorted_02))))),np.power(10,float(math.ceil(np.log10(max(randomsample_sorted_97))))))

#Make log.
mpl.loglog()

#Append graph title.
mpl.title('Significance Determination, Visualization')


##Finally, make the plot visible.
mpl.show()
