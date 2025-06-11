# Flask is framework used to connect Python and HTML
from flask import Flask, render_template, request

# operating system library to get current folder
import os

# classic Python packages
import numpy as np
import matplotlib.pyplot as plt # plotting libraries

# introductory commands
app = Flask(__name__)
app.config["DEBUG"] = True

current_dir = os.path.abspath(os.path.dirname(__file__))
outputName = os.path.join(current_dir, 'static', 'wealth.png')

currVol = 10.89 # Volatility of May 2025, VIX daily average
currRate = 6.29 # BAA rate May 2025, daily average

DIM = 5 # number of white noise dimensions: independent identically distributed
NSIMS = 1000 # number of simulations
NDISPLAYS = 5 # number of displayed graphs in Monte Carlo simulations

alpha = 0.847850 # intercept for AR(1) log S&P volatility, annual
beta = 0.620146 # slope for AR(1) log S&P volatility, annual
gamma = 0.427879 # intercept for AR(1) BAA rate, average daily December
theta = 0.937948 # slope for AR(1) BAA rate, average daily December
cUS = 22.529302 # intercept for S&P US stock returns
volUS = 1.004078 # volatility coefficient for S&P US stock returns
durUS = 6.755580 # duration coefficient for S&P US stock returns
cIntl = 29.408882 # intercept for international stock returns
volIntl = 1.892827 # volatility coefficient for international stock returns
durIntl = 4.905754 # duration coefficient for international stock returns
cBond = -1.104258 # intercept for corporate bond returns
durBond = 6.040179 # duration coefficient for corporate bond returns

# covariance matrix for regression residuals
covMatrix = [[0.132753, -0.065028, -0.043977, 0.106789, 0.129096], [-0.065028, 2.82715, 0.98068, -0.05809, 1.84788], [-0.043977, 0.98068, 3.979577, 0.027882, -0.281541], [0.106789, -0.05809, 0.027882, 1.004464, 0.064041], [0.129096, 1.84788, -0.281541, 0.064041, 8.115026]]

# simulate portfolio returns
# first two arguments are initial volatility and rate
# nYears is the number of years
# last two arguments are shares of bonds in portfolio
# and international among stocks
def simReturns(initVol, initRate, nYears, bondPstart, bondPend, intlP):

    # initializing array simulations
    simLVol = np.zeros((NSIMS, nYears + 1))
    simLVol[:, 0] = np.log(initVol) * np.ones(NSIMS)
    simRate = np.zeros((NSIMS, nYears + 1))
    simRate[:, 0] = initRate * np.ones(NSIMS)
    simUS = np.zeros((NSIMS, nYears))
    simIntl = np.zeros((NSIMS, nYears))
    simBond = np.zeros((NSIMS, nYears))

    # simulate multivariate normal residuals with given covariance matrix
    # need this to simulate these random terms fast
    simResiduals = np.random.multivariate_normal(np.zeros(DIM), covMatrix, (NSIMS, nYears))

    noiseVol = simResiduals[:, :, 0] # noise for autoregression for volatility
    noiseUS = simResiduals[:, :, 1] # noise for US returns
    noiseIntl = simResiduals[:, :, 2] # noise for international returns
    noiseRate = simResiduals[:, :, 3] # noise for autoregression for rate
    noiseBond = simResiduals[:, :, 4] # noise for bond returns

    # simulate log volatility and rate as autoregression
    for t in range(nYears):
        simLVol[:, t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[:, t] + noiseVol[:, t]
        simRate[:, t + 1] = gamma * np.ones(NSIMS) + theta * simRate[:, t] + noiseRate[:, t]

    simD = simRate[:, 1:] - simRate[:, :-1] # one-year change in simulated rate
    simVol = np.exp(simLVol) # from log volatility to volatility

    # simulate US S&P stock, international stock, and corporate bond returns
    # as linear regressions vs simulated factors
    simUS = cUS * np.ones((NSIMS, nYears)) - volUS * simVol[:, 1:] - durUS * simD + noiseUS * simVol[:, 1:]
    simIntl = cIntl * np.ones((NSIMS, nYears)) - volIntl * simVol[:, 1:] - durIntl * simD + noiseIntl * simVol[:, 1:]
    simBond = simRate[:, :-1] + cBond * np.ones((NSIMS, nYears)) - durBond * simD + noiseBond

    simStock = simIntl * intlP + simUS * (1 - intlP) # simulate stock portfolio
    bondP = np.linspace(bondPstart, bondPend, nYears)
    stockP = np.ones(nYears) - bondP
    simOverall = simBond * bondP + simStock * stockP # simulate combined stock and bond portfolio
    return simOverall/100 # returns were arithmetic, in percentages, so need to divide by 100

# simulate wealth process given initial wealth and contributions/withdrawals
# some arguments are the same as in the previous function
# others are: initialW = initialWealth and
# initialFlow (signed value) = first year flow: contribution (+) or withdrawal (-)
# growthFlow (signed value) = annual growth (+) or decline (-) of flow
def simWealth(initVol, initRate, initialW, initialFlow, growthFlow, nYears, bondShare0, bondShare1, intlShare):

    #simulate returns of this portfolio
    simRet = simReturns(initVol, initRate, nYears, bondShare0, bondShare1, intlShare)
    timeAvgRet = np.mean(simRet, axis = 1) # average returns over each simulation
    wealth = np.zeros((NSIMS, nYears + 1)) # create an array for wealth simulation
    wealth[:, 0] = np.ones(NSIMS) * initialW # initial wealth year 0 initialize

    # create (deterministic) array for flow (contributions and withdrawals)
    # for each year in nYears, exponentially growing or decreasing
    flow = initialFlow * np.exp(np.array(range(nYears)) * np.log(1 +  growthFlow))

    # this is the main function connecting wealth to returns and flow
    # we stop simulations for this path if wealth drops below zero
    # this is the only place when we use loop over all simulations
    # if only flow is negative, so there are withdrawals which can make bankrupt
    if initialFlow < 0:
        for t in range(nYears):
            # main equation connecting returns, flow, wealth
            wealth[:, t+1] = np.maximum(wealth[:, t] * (1 + simRet[:, t]) + flow[t] * np.ones(NSIMS), 0)
    else: # if no withdrawals then we do not need to check for bankruptcy
        for t in range(nYears):
            # main equation connecting returns, flow, wealth
            wealth[:, t+1] = wealth[:, t] * (1 + simRet[:, t]) + flow[t] * np.ones(NSIMS)

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
def setupText(initialWealth, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare):

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
                growthText = ' annual increase ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = ' annual decrease ' + percent(abs(growthFlow))
            flowText = initialFlowText + growthText

    # case when withdrawals
    if initialFlow < 0:
        initFlow = form(abs(initialFlow))
        if growthFlow == 0: # no change in withdrawals from year to year
            flowText = 'Constant withdrawals ' + initFlow
        else:
            initialFlowText = 'Initial withdrawals ' + initFlow
            if growthFlow > 0:
                growthText = ' annual increase ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = ' annual decrease ' + percent(abs(growthFlow))
            flowText = initialFlowText + growthText

    # text output explaining portfolio weights
    # for example 33% American: S&P 500 and 67% International: MSCI EAFE
    usText = 'Stocks: ' + percent(1 - intlShare) + ' American: S&P 500'
    stockPortfolioText = usText + '\nand ' + percent(intlShare) + ' International: MSCI EAFE'
    initPortfolioText = 'Portfolio: Stocks and US investment-grade bonds\n'
    bondPortfolioText = 'From ' + percent(1 - bondShare0) + ' Stocks ' + percent(bondShare0) + ' Bonds\nTo ' + percent(1 - bondShare1) + ' Stocks ' + percent(bondShare1) + ' Bonds\n'
    portfolioText = initPortfolioText + bondPortfolioText + stockPortfolioText

    # number of simulations, convert to string
    simText = str(NSIMS) + ' Monte Carlo simulations'

    # number of years in time horizon
    timeHorizonText = 'Time Horizon: ' + str(timeHorizon) + ' years'

    # initial wealth
    initWealthText = 'Initial Wealth ' + form(initialWealth)

    # return all these texts combined
    texts = [simText, portfolioText, timeHorizonText, initWealthText, flowText]

    # combine all these texts and return combined text
    SetupText = 'SETUP: '
    for text in texts:
        SetupText = SetupText + text + '\n'
    return SetupText

# Create simulations and draw them on a picture
# select 5 paths corresponding 10%, 30%, 50%, 70%, 90% ranked by final wealth
# this includes paths which end in zero wealth (ruin, bankruptcy)
# and write a legend for each path, and the overall legend
# including setup in the above function and the results
# the arguments are the same as for the function 'simWealth'
# except initial volatility = current volatility for 2024
def output(initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare):

    # simulate wealth process of a portfolio
    timeAvgRet, paths = simWealth(currVol, currRate, initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare)

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
    SetupText = setupText(initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare)

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
    plt.savefig(outputName, bbox_inches = 'tight')
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

    # initial and terminal share of bonds in portfolio, converted from %
    bond0 = float(request.form['bond0'])*0.01
    bond1 = float(request.form['bond1'])*0.01
    # share of international among stocks, converted from %
    intl = float(request.form['intl'])*0.01

    # number of years and initial wealth for simulation
    nYears = int(request.form['years'])
    initialWealth = float(request.form['initWealth'])

    # Do you withdraw = '-1' or contribute = '+1' annually?
    action = int(request.form.get('action'))
    # Do you annually increase = '+1' or decrease = '-1' these amounts?
    change = int(request.form.get('change'))

    # First year contributions or withdrawals
    initialFlow = float(request.form['initialFlow'])*action

    # Annual change amount for withdrawals or contributions converted from %
    growthFlow = float(request.form['growthFlow'])*0.01*change

    # Draw the PNG picture with simulation results and graphs
    output(initialWealth, initialFlow, growthFlow, nYears, bond0, bond1, intl)

    # the response page after clicking Submit, with this PNG picture
    return render_template('response_page.html')
