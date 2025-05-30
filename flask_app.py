# Flask is framework used to connect Python and HTML
from flask import Flask, render_template, request

# operating system library to get current folder
import os

# classic Python packages
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt # plotting libraries
from statsmodels.api import OLS # ordinary least squares for linear regression

# introductory commands
app = Flask(__name__)
app.config["DEBUG"] = True

# name of current folder
current_dir = os.path.abspath(os.path.dirname(__file__))

# This is reading and preprocessing data
# Due to importance, we have global variables here
# We do not format this as a separate function for this reason

# Reading and prerocessing the main data file
dataFile = os.path.join(current_dir, 'static', 'annual.xlsx')
data = pd.read_excel(dataFile, sheet_name = 'data')
vol = data['Volatility'].values[1:]

# number of data points for the main data file: 1929-2024
# number of data points for residuals of autoregression for log volatility
NDATA = len(vol) - 1
price = data['Price'].values # nominal level of S&P
dividend = data['Dividends'].values[1:] # nominal annual dividend of S&P
cpi = data['CPI'].values # consumer price index for inflation adjustments

# total nominal returns for S&P
nominalReturns = np.log(price[1:] + dividend) - np.log(price[:-1])

# reading and preprocessing the file for international returns
intlFile = os.path.join(current_dir, 'static', 'international.xlsx')
intl = pd.read_excel(intlFile)
# international nominal total returns
nominalIntlRet = intl['International'].values
NINTL = len(nominalIntlRet) # length of data for international returns

currentVol = vol[-1] # volatility of 2024

DIM = 3 # number of white noise dimensions: independent identically distributed
NSIMS = 2500 # number of simulations
NDISPLAYS = 5 # number of displayed graphs in Monte Carlo simulations

# fitting the financial data
# we return series of simulated innovations and model coefficients
# innovations are simulated as multivariate normal
# for that, we use empirical covariance matrix of regression residuals
# infl = 'N' or 'R' for nominal or real
def fitData(infl, nYears):
    # fitting autoregression for log volatility
    lvol = np.log(vol)

    # autoregression of order 1 for linear regression
    Reg = stats.linregress(lvol[:-1], lvol[1:])
    alphaVol = Reg.intercept # intercept of this autoregression
    betaVol = Reg.slope # slope for this autoregression

    # residuals of autoregression
    residVol = lvol[1:] - betaVol * lvol[:-1] - alphaVol * np.ones(NDATA)

    # here we compute US and International total returns, nominal or real
    if infl == 'R':
        inflation = np.diff(np.log(cpi))
        USReturns = nominalReturns - inflation
        intlReturns = nominalIntlRet - inflation[-NINTL:]
    if infl == 'N':
        USReturns = nominalReturns
        intlReturns = nominalIntlRet

    # normalized (divided by volatility) US and international returns
    normUSReturns = USReturns/vol
    normIntlReturns = intlReturns/vol[-NINTL:]

    # empirical mean of US returns
    meanUSReturns = np.mean(normUSReturns)

    # regression of international upon US returns after normalization
    DFreg = pd.DataFrame({'const' : 1/vol[-NINTL:], 'vol' : 1, 'benchmark' : normUSReturns[-NINTL:]})
    Reg = OLS(normIntlReturns, DFreg).fit()
    regResiduals = Reg.resid

    # computation of covariance matrix for series of empirical residuals
    covVolUS = np.cov(residVol, normUSReturns[1:]) # use all data 1929-2024
    # use data 1970-2024
    allResids = np.stack((residVol[-NINTL:], normUSReturns[-NINTL:], regResiduals))
    covAll = np.cov(allResids)
    #superimpose covariance from 1929-2024 upon covariance from 1970-2024
    covAll[:2, :2] = covVolUS

    # dictionary containing all coefficients of the model
    # slope and intercept of autoregression for log volatility
    # mean of normalized US returns
    # and regression coefficients for international vs US normalized returns
    fittedCoeffs = dict()
    fittedCoeffs['alphaVol'] = alphaVol
    fittedCoeffs['betaVol'] = betaVol
    fittedCoeffs['meanReturns'] = meanUSReturns
    fittedCoeffs.update(Reg.params)

    # simulation of series of innovations as multivariate normal
    # with mean zero and empirical covariance matrix
    # there are NSIMS x nYears such independent simulated vectors
    # so we simulate in advance enough for Monte Carlo simulations
    # such simulations are done in the next function
    # This united simulation was done to increase speed
    # since loops in Python are slower than using numpy which uses C
    simResiduals = np.random.multivariate_normal(np.zeros(DIM), covAll, (NSIMS, nYears))

    # return two values: dictionary of statistically fitted parameters
    # and simulated
    return fittedCoeffs, simResiduals

# Monte Carlo simulations of portfolio returns using the previous function
# infl = Adjust for inflation? 'R' = Yes, 'N' = No
# initialVol = initial annual volatility for simulation
# nYears = number of years for simulation
# intlStocks = share of international stocks in portfolio, for example 0.33
def simulation(infl, initialVol, nYears, intlStocks):

    # fit the empirical financial data
    fittedCoeffs, simResiduals = fitData(infl, nYears)

    # return coefficients of this model
    alpha = fittedCoeffs['alphaVol']
    beta = fittedCoeffs['betaVol']
    avg = fittedCoeffs['meanReturns']
    interceptCoeff = fittedCoeffs['const']
    volCoeff = fittedCoeffs['vol']
    usCoeff = fittedCoeffs['benchmark']

    # return three components of the simulated noise
    noiseVol = simResiduals[:, :, 0] # noise for autoregression for volatility
    noiseUS = simResiduals[:, :, 1] # noise for US returns
    noiseIntl = simResiduals[:, :, 2] # noise for regression of international vs US

    # simulate volatility
    simLVol = np.zeros((NSIMS, nYears + 1)) # create log volatility array
    simLVol[:, 0] = np.ones(NSIMS) * np.log(initialVol) # initial step year 0

    # make one autoregression time step simultaneously for all simulations
    # to speed up using numpy instead of loop over all NSIMS simulations
    for t in range(nYears):
        simLVol[:, t+1] = alpha + beta * simLVol[:, t] + noiseVol[:, t]
    simVol = np.exp(simLVol) # switch from log volatility to volatility

    # simulate US returns: IID Gaussian from second series of innovations
    # times volatility simulated above
    simUS = simVol[:, 1:] * (np.ones((NSIMS, nYears)) * avg + noiseUS)

    # simulate international returns as regression result
    # using simulated US returns as factor and simulated volatility above
    intercepts = interceptCoeff * np.ones((NSIMS, nYears))
    noiseTermIntl = noiseIntl * simVol[:, 1:]
    simIntl = intercepts + volCoeff * simVol[:, 1:] + usCoeff * simUS + noiseTermIntl

    # simulate and return annual portfolio returns
    # as a constant linear combination of US returns and international returns
    simPortfolio = intlStocks * simIntl + (1 - intlStocks) * simUS
    return simPortfolio

# simulate wealth process given initial wealth and contributions/withdrawals
# four arguments are the same as in the previous function
# infl = Adjust for inflation? 'R' = Yes, 'N' = No
# initialVol = initial annual volatility for simulation
# nYears = number of years for simulation
# intlStocks = share of international stocks in portfolio, for example 0.33
# other three are: initialW = initialWealth and
# initialFlow (signed value) = first year flow: contribution (+) or withdrawal (-)
# growthFlow (signed value) = annual growth (+) or decline (-) of flow
def simWealth(infl, initialVol, initialW, initialFlow, growthFlow, nYears, portfolioUS):

    #simulate returns of this portfolio
    simRet = simulation(infl, initialVol, nYears, portfolioUS)
    timeAvgRet = np.mean(simRet, axis = 1) # average returns over each simulation
    wealth = np.zeros((NSIMS, nYears+1)) # create an array for wealth simulation
    wealth[:, 0] = np.ones(NSIMS) * initialW # initial wealth year 0 initialize

    # create (deterministic) array for flow (contributions and withdrawals)
    # for each year in nYears, exponentially growing or decreasing
    flow = initialFlow * np.exp(np.array(range(nYears)) * np.log(1 +  growthFlow))

    # this is the main function connecting wealth to returns and flow
    # we stop simulations for this path if wealth drops below zero
    # this is the only place when we use loop over all simulations
    # if only flow is negative, so there are withdrawals which can make bankrupt
    if initialFlow <= 0:
        for sim in range(NSIMS):
            for t in range(nYears):
                if (wealth[sim, t] < 0): # if wealth became negative
                    wealth[sim, t] = 0 # we stop simulating this path
                    break # and the current and future wealth is zero
                else:
                    # main equation connecting returns, flow, wealth
                    wealth[sim, t+1] = wealth[sim, t] * np.exp(simRet[sim, t]) + flow[t]
    else: # if no withdrawals then we do not need to check for bankruptcy
        for t in range(nYears):
            # main equation connecting returns, flow, wealth
            wealth[:, t + 1] = wealth[:, t] * np.exp(simRet[:, t]) + flow[t]

    # timeAvgRet = average total portfolio return array over each path
    # wealth = paths of wealth
    return timeAvgRet, wealth

# Percentage format for probability 'x' rounded to 2 decimal points
# for text in output picture legend say 45.33%
def percent(x):
    return str(round(100*x, 2)) + '%'

# Wealth amount format with K, M, B and one decimal point
# to simplify output and make legend less cluttered
# K = 1,000, M = 1,000,000, B = 1,000,000,000
def form(x):
    if x < 10**3:
        return f"{x:.1f}"
    if 10**3 <= x < 10**6: # 1.2K not 1236
        return f"{10**(-3)*x:.1f}K"
    if 10**6 <= x < 10**9: # 15.2M not 15192124
        return f"{10**(-6)*x:.1f}M"
    if 10**9 <= x: # 24.7B not 24694M
        return f"{10**(-9)*x:.1f}B"

# Vertical lines on the graph of simulations
def allTicks(horizon):
    if horizon < 10:
        return range(horizon + 1) # if less than 10 years make all lines visible
    else: # make a line visible every 5 years, including the start
        step = int(horizon/5) # how many lines with 5-year intervals
        if horizon - 5 * step > 2: # horizon = 14, then lines = 0, 5, 10, 14
            return np.append(np.array(range(6))*step, [horizon])
        else: # horizon = 12, then lines = 0, 5, 12
            return np.append(np.array(range(5))*step, [horizon])

# text for legend: setup part, where we explain in words the inputs
# output will be created after simulation in the next function
# need to print this in the legend to the right of the main picture
# to remind the investor about their inputs
# the arguments are the same as for 'simWealth' except initial volatility
# which is here always assumed to be current volatility
# for 2024, the last year available as of this writing
def setupText(infl, initialWealth, initialFlow, growthFlow, timeHorizon, intlShare):

    # This part is text description of flow (contributions or withdrawals)
    # Initial value for year 1 and rate of annual increase/decrease
    if initialFlow == 0:
        flowText = 'No regular contributions or withdrawals'

    # case when contributions
    if initialFlow > 0:
        initFlow = form(initialFlow)
        if growthFlow == 0: # no change in contributions from year to year
            flowText = 'Constant contributions ' + initFlow
        else:
            initialFlowText = 'Initial contributions ' + initFlow
            if growthFlow > 0:
                growthText = 'Annual increase ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'Annual decrease ' + percent(abs(growthFlow))
            flowText = initialFlowText + '\n' + growthText

    # case when withdrawals
    if initialFlow < 0:
        initFlow = form(abs(initialFlow))
        if growthFlow == 0: # no change in withdrawals from year to year
            flowText = 'Constant withdrawals ' + initFlow
        else:
            initialFlowText = 'Initial withdrawals ' + initFlow
            if growthFlow > 0:
                growthText = 'Annual increase ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'Annual decrease ' + percent(abs(growthFlow))
            flowText = initialFlowText + '\n' + growthText

    # text output explaining portfolio weights
    # for example 33% American: S&P 500 and 67% International: MSCI EAFE
    portfolioText = 'Stocks: ' + percent(1 - intlShare) + ' American: S&P 500\nand ' + percent(intlShare) + ' International: MSCI EAFE'

    # number of simulations, convert to string
    simText = str(NSIMS) + ' Monte Carlo simulations'

    # number of years in time horizon
    timeHorizonText = 'Time Horizon: ' + str(timeHorizon) + ' years'

    # inflation choice: Adjust or not?
    if infl == 'N':
        inflText = 'Nominal, not inflation-adjusted'
    if infl == 'R':
        inflText = 'Real, inflation-adjusted'

    # initial wealth
    initWealthText = 'Initial Wealth ' + form(initialWealth)

    # return all these texts combined
    texts = [simText, portfolioText, timeHorizonText, inflText, initWealthText, flowText]

    # combine all these texts and return combined text
    SetupText = 'SETUP: '
    for text in texts:
        SetupText = SetupText + '\n' + text
    return SetupText

# Create simulations and draw them on a picture
# select 5 paths corresponding 10%, 30%, 50%, 70%, 90% ranked by final wealth
# this includes paths which end in zero wealth (ruin, bankruptcy)
# and write a legend for each path, and the overall legend
# including setup in the above function and the results
# the arguments are the same as for the function 'simWealth'
# except initial volatility = current volatility for 2024

def output(infl, initialW, initialFlow, growthFlow, timeHorizon, intlShare):

    # simulate wealth process of a portfolio
    timeAvgRet, paths = simWealth(infl, currentVol, initialW, initialFlow, growthFlow, timeHorizon, intlShare)

    # take average total portfolio return over each path
    # and pick paths which do not end in bankruptcy (ruin)
    # average these averaged return over such paths
    avgRet = np.mean([timeAvgRet[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1]) # average final wealth over paths

    # share of paths which end in bankruptcy (ruin) = zero wealth
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])

    # sort all paths by final wealth from bottom to top
    sortedIndices = np.argsort(paths[:, -1])

    # indices for selected paths ranked by final wealth
    # NDISPLAYS = number of displayed paths on the main image
    # equidistant by ranks of final wealth
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]

    # all time points: 0, 1, ..., timeHorizon
    times = range(timeHorizon + 1)
    if np.isnan(avgRet): # all paths end in ruin, all have zero final wealth
        ResultText = 'RESULTS: 100% Ruin Probability'
    else: # simulation results
        RuinProbText = 'Ruin Probability ' + percent(ruinProb)
        AvgRetText = 'average returns of paths without ruin ' + percent(avgRet)
        MeanText = 'average final wealth ' + form(wealthMean)
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText

    # text for setup which is in the main legend for the plot
    # so that user sees output image in a different page than inputs
    # and does not forget these inputs
    SetupText = setupText(infl, initialW, initialFlow, growthFlow, timeHorizon, intlShare)

    # this plot of only one point in white color is necessary for big legend
    # because it serves as its anchor
    plt.plot([0], [initialW], color = 'w', label = SetupText + '\n' + ResultText)

    # next show plots of wealth paths
    for display in range(NDISPLAYS):
        index = selectedIndices[display]

        # text shows final wealth and its % rank
        rankText = ' final wealth, ranked ' + percent((2*display + 1)/(2*NDISPLAYS))
        endWealth = paths[index, -1]

        if (endWealth == 0): # this path ended with zero wealth
            pathLabel = '0' + rankText + ' Gone Bust !!!'
        else: # this path ends with positive wealth
            pathLabel = form(endWealth) + rankText + ' returns: ' + percent(timeAvgRet[index])

        plt.plot(times, paths[index], label = pathLabel)

    plt.gca().set_facecolor('ivory') # background plot color
    plt.xlabel('Years') # label of the X-axis, for time

    # make vertical lines selected years
    ticks = allTicks(timeHorizon)
    plt.xticks(ticks)

    plt.title('Wealth Plot') # title of the entire figure

    # properties of legend: location relative to the anchor above
    # font size and background color
    plt.legend(bbox_to_anchor=(1, 1.04), loc='upper left', prop={'size': 14}, facecolor = 'azure')
    plt.grid(True) # make vertical and horizontal grid

    # save to folder 'static' to present in output page below
    image_path = os.path.join(current_dir, 'static', 'wealth.png')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

# main landing page
@app.route('/')
def index():
    if request.method == "GET":
        return render_template("main_page.html")

# main function executing when click Submit
# differs from the main landing page by /compute
@app.route('/compute', methods=["GET", "POST"])
def compute():

    # share of international stocks in your portfolio, converted from %
    intl = float(request.form['intl'])*0.01

    # number of years and initial wealth for simulation
    nYears = int(request.form['years'])
    initialWealth = float(request.form['initWealth'])

    # Do you adjust for inflation? Yes = 'R', No = 'N'
    inflAdjusted = request.form.get('adjusted')

    # Do you withdraw = '-1' or contribute = '+1' annually?
    action = int(request.form.get('action'))

    # Do you annually increase = '+1' or decrease = '-1' these amounts?
    change = int(request.form.get('change'))

    # First year contributions or withdrawals
    initialFlow = float(request.form['initialFlow'])*action

    # Annual change amount for withdrawals or contributions converted from %
    growthFlow = float(request.form['growthFlow'])*0.01*change

    # Draw the PNG picture with simulation results and graphs
    output(inflAdjusted, initialWealth, initialFlow, growthFlow, nYears, intl)

    # the response page after clicking Submit, with this PNG picture
    return render_template('response_page.html')