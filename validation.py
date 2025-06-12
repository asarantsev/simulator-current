import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot
from statsmodels.api import OLS
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

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
def analysis(noise):
    print('skew = ', stats.skew(noise))
    print('kurtosis = ', stats.kurtosis(noise))
    print('Shapiro-Wilk p = ', stats.shapiro(noise)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(noise)[1])
    print('L1 = ', sum(abs(acf(noise, nlags = 5)[1:])))
    print('L1 = ', sum(abs(acf(abs(noise), nlags = 5)[1:])))
    
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
DIM = 5 # number of regressions

# print output of regressions
for k in range(DIM):
    print(allNames[k], '\n') # name of regression
    regression = allRegressions[k] # regression itself
    print(regression.summary()) # print regression summary
    print('coefficients')
    print(regression.params) # print regression parameters
    resids = regression.resid.values # residuals of this regression
    allResiduals.append(resids) # write them to compute covariance matrix later
    plots(resids, allNames[k]) # normality and autocorrelation function plots
    analysis(resids) # are these residuals normal white noise?

# computation of the empirical covariance matrix
# initialize by zeros
covMatrix = [[0 for item in range(DIM)] for item in range(DIM)]
for k in range(DIM):
    for l in range(DIM):
        if k == l:
            covMatrix[k][k] = round(np.var(allResiduals[k]), 6)
        else:
            Q = min(len(allResiduals[k]), len(allResiduals[l]))
            covMatrix[k][l] = round(np.cov(allResiduals[k][-Q:], allResiduals[l][-Q:])[0][1], 6)

print('empirical covariance matrix')
print(covMatrix)