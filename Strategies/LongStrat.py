from LOTlib3.Miscellaneous import q, random
from LOTlib3.Grammar import Grammar
from LOTlib3.DataAndObjects import FunctionData, Obj
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Eval import primitive
from LOTlib3.Miscellaneous import qq
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
from multiprocessing import Pool
from Engine.TradingSim import TradeSim
from NewFeatures.CreateFeatures import detect_candle_patterns
from math import log
import random
import pandas as pd
import numpy as np
import time
import sys
# import os

# ─── 1. Primitives and Helper Functions ──────────────
def parse_shorthand(s):
    if isinstance(s, (int, float)):
        return s  # Already numeric

    s = str(s).strip().upper()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}

    if s[-1] in multipliers:
        return float(s[:-1]) * multipliers[s[-1]]
    else:
        return float(s)

### Basic Primitives

@primitive
def tuple_(A, B, C, D):
    return (A, B, C, D)

@primitive
def getp_(x, i, p, var):
    if p != i:
        return x.iloc[i-p][var]
    return x.iloc[i][var]

@primitive
def get_(x, i, var):
    return x.iloc[i][var]

@primitive
def g_(x,y): return x>y

@primitive
def ge_(x,y): return x>=y

@primitive
def l_(x,y): return x<y

@primitive
def le_(x,y): return x<=y

@primitive
def e_(x,y): return x==y

### Advanced Primitives

# @primitive
# def

# ─── 2. Define Grammar for boolean entry/exit rules ──────────────
grammar = Grammar(start='Strategy')

# A rule is a comparison: lt_(x.RSI, 30)
grammar.add_rule('Strategy', 'tuple_', ['Entry', 'Exit', 'TradeSize', 'SL'], 1.0)
grammar.add_rule('Entry', '', ['Cond'], 1.0)
grammar.add_rule('Exit', '', ['Cond'], 1.0)
grammar.add_rule('TradeSize', '', ['Size'], 1.0)

# Trade Size constants
#for tSize in range(0, 100, 10):
grammar.add_rule('Size', str(1), None, 1.0)

grammar.add_rule('SL', '0', None, 1.0)

for sloss in np.arange(0.5, 6, 0.5):
    grammar.add_rule('SL', str(sloss), None, 0.6)

# # add not into conditions
grammar.add_rule('nCond', '', ['Cond'], 1.0)
grammar.add_rule('nCond', 'not_', ['Cond'], 0.8)

# add two simple boolean conditions
grammar.add_rule('Cond', '', ['nComp'], 1.0)
grammar.add_rule('Cond', 'and_', ['nComp', 'nComp'], 1.0)
grammar.add_rule('Cond', 'and_', ['nComp', 'nCond'], 1.0)
grammar.add_rule('Cond', 'and_', ['nCond', 'nCond'], 0.8)
grammar.add_rule('Cond', 'or_', ['nComp', 'nComp'], 1.0)
grammar.add_rule('Cond', 'or_', ['nComp', 'nCond'], 1.0)
grammar.add_rule('Cond', 'or_', ['nCond', 'nCond'], 0.8)

# add the comparison of values by operator
grammar.add_rule('nComp', '', ['Comp'], 1.0)
grammar.add_rule('nComp', 'not_', ['Comp'], 0.8)

# add rules to get candlestick patterns (boolean features)
grammar.add_rule('Comp', 'get_', ["x", "i", "CPat"], 1.0)
grammar.add_rule('Comp', 'getp_', ["x", "i", 'Const', 'CPat'], 1.0)

# add rule to select a specific candlestick pattern
for candlePat in ['"green"', '"red"', '"bullish_engulfing"', '"bearish_engulfing"', '"morning_star"', '"evening_star"', '"hammer"', 
                  '"hanging_man"', '"inverted_hammer"', '"shooting_star"', '"dragonfly_doji"', '"gravestone_doji"', '"standard_doji"']:
    grammar.add_rule('CPat', candlePat, None, 1.0)

# Add the different operators
for op in ['g_', 'ge_', 'l_', 'le_', 'e_']:
    grammar.add_rule('Comp', op, ['GetIndicator', 'Const'], 1.0)
    grammar.add_rule('Comp', op, ['GetIndicator', 'GetIndicator'], 1.0)
    grammar.add_rule('Comp', op, ['GetIndicator', 'GetHIndicator'], 1.0)
    grammar.add_rule('Comp', op, ['GetPrice', 'GetHPrice'], 1.0)
    #grammar.add_rule('Comp', op, ['GetPrice', 'GetPrice'], 1.0)

# create price retrieval rules
grammar.add_rule('GetPrice', 'get_', ["x", "i", 'Price'], 1.0)
grammar.add_rule('GetHPrice', 'getp_', ["x", "i", 'Const', 'Price'], 1.0)

# create indicator retrieval rules
grammar.add_rule('GetIndicator', 'get_', ["x", "i", 'Indicator'], 1.0)
grammar.add_rule('GetHIndicator', 'getp_', ["x", "i", 'Const', 'Indicator'], 1.0)

# all price variables
for p in ['"close"', '"open"', '"low"', '"high"']:
    grammar.add_rule('Price', p, None, 1.0)

# all indicator variables
for i in ['"RSI"', '"RSI-based MA"', '"Histogram"', '"MACD"', '"Signal"' , '"ADX"']:
    grammar.add_rule('Indicator', i, None, 1.0)

# Numeric constants used in conditions
for const in range(1, 100):
    grammar.add_rule('Const', str(const), None, 1 / const)


# ─── 2. Custom compound hypothesis for entry + exit rules ─────────

class TradingStrategy(LOTHypothesis):
    def __init__(self, **kwargs):
        super().__init__(grammar=grammar, **kwargs)
        self.trades = []  # to store trades


    def compute_single_likelihood(self, data):
        self.trades = []  # reset for fresh evaluation

        # The actual data rows (as a DataFrame)
        priceData = data.input[0]

        # store the current strategy
        code_str = str(self)  # LOTlib3's compiled lambda
        raw_expr = code_str[len("lambda x:"):].strip()
        program = eval(f"lambda x, i: {raw_expr}")

        # perform the trade simulator to check profitability of the current strategy
        trades = TradeSim(program, priceData)

        # keep the trade data
        self.trades = trades

        if len(trades) == 0:
            return -np.inf  # harsh penalty for not trading

        # make the traeds into a dataframe
        dftrades = pd.DataFrame(trades)

        # compute sharpe parameter to reduce risk due to volatile strategies
        #returns = dftrades['pct_change']
        #sharpe = returns.mean() / (returns.std() + 1e-6)

        # compute final strategy fitness
        fitness = dftrades['profit'].sum() #+ sharpe

        # return log product
        return fitness + 1e-5
        #np.log(max(fitness + 1e-5, 1e-5)) * 20         

# ─── 3. Parallel model run functions ─────────
def run_chain(seed_data_h0):
    # unpack and set seed
    seed, data, h0, steps = seed_data_h0
    random.seed(seed)

    # create a top 10 variable
    top = TopN(N=10)

    # only report the step and time taken for the first chain during the exploration
    if seed == 1000:
        startTime = time.time()
        tell = range(0, steps, int(steps/10))

        # generate and evaluate hypothesis with a MCMC
        for i, h in enumerate(MetropolisHastingsSampler(h0, data, steps=steps)):
            top << h
            
            # report the step and time
            if i in tell:
                stepTime = time.time()
                print("Step: ", i, "Time: ", stepTime - startTime)
    
    else:            
        # generate and evaluate hypothesis with a MCMC
        for i, h in enumerate(MetropolisHastingsSampler(h0, data, steps=steps)):
            top << h

    return top

# --- Main function to run chains in parallel ---
def parexplore_search(h0, data, totalSteps, n_chains=10, top_k=10):
    # wrap data in jobs with seeds for reproducilibity
    seeds = [1000 + i for i in range(n_chains)]
    jobs = [(seed, data, h0, int(totalSteps/n_chains)) for seed in seeds]

    with Pool(processes=n_chains) as pool:
        topn_lists = pool.map(run_chain, jobs)

    # Merge all top-k into a global TopN
    global_top = TopN(N=top_k)
    for local_top in topn_lists:
        for h in local_top:
            global_top << h

    return global_top

if __name__ == '__main__':
    # get the relevant time frame
    timeFrame = sys.argv[2]

    # get the total steps the MCMC has to take
    Steps = sys.argv[1]
    TotalSteps = parse_shorthand(Steps)

    # load in data
    data = pd.read_csv("Data/PriceData/KRAKEN_ADAUSD, " + timeFrame + ".csv")
    
    # change time
    data['time'] = pd.to_datetime(data['time'], utc=True)
    
    # add candle features
    data = pd.concat([data, detect_candle_patterns(data)], axis=1)

    # store the data as a functionData object for the MCMC
    wdata = [FunctionData(input=[data], output=None, alpha=0.95)]

    # create the compound hypothesis
    h0 = TradingStrategy()

    # track time
    start = time.time()
 
    # run the parallel symbolic strategy explorer
    tn = parexplore_search(h0, wdata, TotalSteps)

    # intialize lists, to extract topN hypothesis data from
    mvData = pd.DataFrame.from_records([{'posterior': h.posterior_score, 'prior': h.prior, 
                                         'likelihood': h.likelihood, 'rule': qq(h)} for h in tn])
    
    # store the trades seperately and add a new Strategy ID column
    trades = [pd.DataFrame({'Strategy' : i, 'Trades': h.trades}) if type(h.trades) == list else h.trades.assign(Strategy = i) for i, h in enumerate(tn)]

    # print results
    print(mvData)

    # change max colwidth so rule is printed completely
    pd.set_option('display.max_colwidth', None)
    print(mvData['rule'])

    # store strategy trade summary 
    summary_rows = []

    # print and store the results of top 10 strategies
    for df in trades:

        if df.empty:
            row = {
                'Strategy' : df['Strategy'],
                'Total Profit (%)': 0,
                'Median % Change': 0,
                'Max % Change': 0,
                'Number of Trades': 0,
                'Win Rate (%)': 0,
                'Avg Holding Time' : pd.Timedelta(0)
            }
        else:
            # add new row to indicate wins.
            df['win'] = df['profit'] > 0

            profit = df['profit'].sum()

            # create the summary dataframe for the current set of trades
            row = {
                'Strategy' : df['Strategy'][0],
                'Total Profit (%)': profit,
                'Median % Change': df['pct_change'].median(),
                'Max % Change': df['pct_change'].max(),
                'Number of Trades': len(df),
                'Win Rate (%)': df['win'].mean() * 100,
                'Avg Holding Time' : df['time-transpired'].mean().round('s')
            }

        # store the row regardless
        summary_rows.append(row)

    # turn it into a dataframe
    summary = pd.DataFrame(summary_rows)

    # flatten the list of trades
    best_tradesdf = pd.concat(trades)

    # save the data in csv
    mvData.to_csv("Data/FoundStrats/LongStrat" + Steps + timeFrame + ".csv")
    best_tradesdf.to_csv("Data/Trades2Vis/LongStrat" + Steps + timeFrame + ".csv")

    # Optionally round
    summary = summary.round(2)

    # Display
    print(summary)

    end = time.time()
    print("Time spend:", end-start)
