import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot
from statsmodels.api import OLS
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# This is a technical command needed to write data in pandas data frames
pd.options.mode.copy_on_write = True 

skewAll = []
kurtAll = []
SWp = []
JBp = []
L1O = []
L1A = []

# Plots for autocorrelation function for original
# and absolute values and for quantile-quantile plot
# versus normal distribution
def plots(data, label):
    plot_acf(data)
    pyplot.title('ACF original ' + label)
    pyplot.savefig(label + '-oacf.png')
    pyplot.close()
    pyplot.show()
    plot_acf(abs(data))
    pyplot.title('ACF absolute ' + label)
    pyplot.savefig(label + '-aacf.png')
    pyplot.close()
    qqplot(data, line = 's')
    pyplot.title('QQ plot ' + label)
    pyplot.savefig(label + '-qq.png')
    pyplot.close()

# Are residuals 'noise' Gaussian and white noise?
def analysis(data):
    skewAll.append(round(stats.skew(data), 3))
    kurtAll.append(round(stats.kurtosis(data), 3))
    SWp.append(round(stats.shapiro(data)[1], 3))
    JBp.append(round(stats.jarque_bera(data)[1], 3))
    L1O.append(round(sum(abs(acf(data, nlags = 5)[1:])), 3))
    L1A.append(round(sum(abs(acf(abs(data), nlags = 5)[1:])), 3))
    
allResiduals = []
allRegressions = []

# read and preprocess data
DF = pd.read_excel('overall.xlsx', sheet_name = None)
main = DF['main']
vol = main['Volatility'].values[1:] # volatility
NVOL = len(vol)
lvol = np.log(vol)
price = main['Price'].values # S&P index
dividend = main['Dividends'].values[1:] # S&P dividend
rate = main['BAA'].values # BAA rate
lrate = np.log(rate)
USReturns = (price[1:] + dividend)/price[:-1] - np.ones(NVOL) # arithmetic returns
normUSReturns = 100 * USReturns/vol # normalized arithmetic returns in %
world = DF['world'] 
intlReturns = world['International'].values # international returns
NINTL = len(intlReturns)
normIntlReturns = 100 * intlReturns/vol[-NINTL:] # normalized intl returns in %
bonds = DF['bonds']
wealthBond = bonds['Bond Wealth'].values
NBOND = len(wealthBond) - 1
bondRet = 100 * np.diff(wealthBond)/wealthBond[:-1] # bond returns in %

# All Regressions 
DF0 = pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})
allRegressions.append(OLS(lvol[1:], DF0).fit()) # AR(1) log volatility
DF1 = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'diff' : np.diff(rate)/vol})
allRegressions.append(OLS(normUSReturns, DF1).fit()) # S&P returns
DF2 = pd.DataFrame({'const' : 1/vol[-NINTL:], 'vol' : 1, 'diff' : np.diff(rate[-NINTL-1:])/vol[-NINTL:]})
allRegressions.append(OLS(normIntlReturns, DF2).fit()) # international returns
DF3 = pd.DataFrame({'const' : 1, 'lag' : lrate[:-1]})
allRegressions.append(OLS(lrate[1:], DF3).fit()) # AR(1) log rate
DF4 = pd.DataFrame({'const' : 1, 'dur' : np.diff(rate[-NBOND-1:])})
allRegressions.append(OLS(bondRet - rate[-NBOND-1:-1], DF4).fit()) # bond returns
# names for these regressions
allNames = ['ln-vol', 'usa-stocks', 'intl-stocks', 'ln-baa', 'bonds']
allResiduals = pd.DataFrame(columns = allNames)
DIM = 5 # number of regressions
lengths = []

# print output of regressions
for k in range(DIM):
    print(allNames[k], '\n') # name of regression
    regression = allRegressions[k] # regression itself
    print(regression.summary()) # print regression summary
    print('coefficients')
    print(regression.params) # print regression parameters
    resids = regression.resid.values # residuals of this regression
    lengths.append(len(resids))
    allResiduals[allNames[k]] = np.pad(resids[::-1], (0, NVOL - lengths[k]), constant_values = np.nan)
    plots(resids, allNames[k]) # normality and autocorrelation function plots
    analysis(resids) # are these residuals normal white noise?

covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print('covariance matrix')
print(covMatrix)
print('correlation matrix')
print(corrMatrix)

statDF = pd.DataFrame({'reg' : allNames, 'skew': skewAll, 'kurt' : kurtAll, 'SW' : SWp, 'JB' : JBp, 'L1O': L1O, 'L1A' : L1A, 'length' : lengths})
print(statDF)

allResiduals.to_excel('innovations.xlsx')