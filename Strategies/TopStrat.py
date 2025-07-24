#import sys
#sys.path.insert(0, 'LOTlib3')
from LOTlib3.Miscellaneous import q, random
from LOTlib3.Grammar import Grammar
from LOTlib3.DataAndObjects import FunctionData, Obj
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Eval import primitive
from LOTlib3.Miscellaneous import qq
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
#from LOTlib3.Primitives.Logic import lt_, lte_, eq_, gt_, gte_
from multiprocessing import Pool
from joblib import Parallel, delayed
from math import log
import random
import pandas as pd
import numpy as np
import time
# import os
# import sys
from Engine.TradingSim import TradeSim

# ─── 1. Primitives and Helper Functions ──────────────

def get_price_pivots(df, span=2):
    """
    Returns boolean lists indicating pivot highs (peaks) and pivot lows (valleys)
    for each row in the DataFrame.

    A pivot high is a local maximum where the high is greater than the highs 
    of `span` bars before and after.
    
    A pivot low is a local minimum where the low is less than the lows 
    of `span` bars before and after.

    Args:
        df (pd.DataFrame): Must contain 'high' and 'low' columns.
        span (int): Number of bars before and after to consider.

    Returns:
        (pivot_high, pivot_low): Tuple of boolean lists of length len(df)
    """
    n = len(df)
    pivot_high = [False] * n
    pivot_low = [False] * n

    for i in range(span, n - span):
        high_window = df['high'].iloc[i - span: i + span + 1]
        low_window = df['low'].iloc[i - span: i + span + 1]

        center_high = df['high'].iloc[i]
        center_low = df['low'].iloc[i]

        if center_high == max(high_window) and list(high_window).count(center_high) == 1:
            pivot_high[i] = True

        if center_low == min(low_window) and list(low_window).count(center_low) == 1:
            pivot_low[i] = True

    return pivot_high, pivot_low

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

for sloss in np.arange(0, 6, 0.5):
    grammar.add_rule('SL', str(sloss), None, 1.0)

# # add not into conditions
grammar.add_rule('nCond', '', ['Cond'], 1.0)
grammar.add_rule('nCond', 'not_', ['Cond'], 0.8)

# add two simple boolean conditions
grammar.add_rule('Cond', '', ['nComp'], 1.0)
grammar.add_rule('Cond', 'and_', ['nComp', 'nComp'], 1.0)
grammar.add_rule('Cond', 'and_', ['nComp', 'nCond'], 0.8)
grammar.add_rule('Cond', 'and_', ['nCond', 'nCond'], 0.2)
grammar.add_rule('Cond', 'or_', ['nComp', 'nComp'], 1.0)
grammar.add_rule('Cond', 'or_', ['nComp', 'nCond'], 0.8)
grammar.add_rule('Cond', 'or_', ['nCond', 'nCond'], 0.2)

# add the comparison of values by operator
grammar.add_rule('nComp', '', ['Comp'], 1.0)
grammar.add_rule('nComp', 'not_', ['Comp'], 0.8)

# Add the different operators
for op in ['g_', 'ge_', 'l_', 'le_', 'e_']:
    grammar.add_rule('Comp', op, ['Get', 'Const'], 1.0)
    grammar.add_rule('Op', op, ['Get', 'Getp'], 1.0)

# get variable rule
grammar.add_rule('Get', 'get_', ["x", "i", 'Var'], 1.0)
grammar.add_rule('Getp', 'getp_', ["x", "i", 'Const', 'Var'], 1.0)

# Available variables from priceData rows
for var in ['"RSI"', '"close"', '"open"', '"low"', '"high"', '"RSI-based MA"', '"Histogram"', '"MACD"', '"Signal"' , '"ADX"']:
    grammar.add_rule('Var', var, None, 1.0)

# Numeric constants used in conditions
for const in range(1, 100):
    grammar.add_rule('Const', str(const), None, 1.0)


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
        returns = dftrades['pct_change']
        sharpe = returns.mean() / (returns.std() + 1e-6)

        # compute final strategy fitness
        fitness = dftrades['profit'].sum() + sharpe

        # return log product
        return np.log(max(fitness + 1e-5, 1e-5))         

# ─── 3. Parallel model run functions ─────────
def run_chain(seed_data_h0):
    # unpack and set seed
    seed, data, h0 = seed_data_h0
    random.seed(seed)

    # create a top 10 variable
    top = TopN(N=10)

    # generate and evaluate hypothesis with a MCMC
    for i, h in enumerate(MetropolisHastingsSampler(h0, data, steps=5000)):
        top << h

    return top

# --- Main function to run chains in parallel ---
def parexplore_search(h0, data, n_chains=10, top_k=10):
    # wrap data in jobs with seeds for reproducilibity
    seeds = [1000 + i for i in range(n_chains)]
    jobs = [(seed, data, h0) for seed in seeds]

    with Pool(processes=n_chains) as pool:
        topn_lists = pool.map(run_chain, jobs)

    # Merge all top-k into a global TopN
    global_top = TopN(N=top_k)
    for local_top in topn_lists:
        for h in local_top:
            global_top << h

    return global_top

if __name__ == '__main__':
    # get the relevant experiment
    #experiment = sys.argv[1]

    # load in data
    data = pd.read_csv("Data/PriceData/KRAKEN_ADAUSD, 15.csv")
    
    # change time
    data['time'] = pd.to_datetime(data['time'], utc=True)

    wdata = [FunctionData(input=[data], output=None, alpha=0.0)]

    # create the compound hypothesis
    h0 = TradingStrategy()

    # track time
    start = time.time()
 
    # run the 
    tn = parexplore_search(h0, wdata)

    # intialize lists, to extract topN hypothesis data from
    mvData = pd.DataFrame.from_records([{'TimeF': '1D', 'posterior': h.posterior_score, 'prior': h.prior, 
                                         'likelihood': h.likelihood, 'rule': qq(h)} for h in tn])
    
    # store the trades seperately
    trades = [h.trades for h in tn]

    # print results
    print(mvData)

    # change max colwidth so rule is printed completely
    pd.set_option('display.max_colwidth', None)
    print(mvData['rule'])

    # store strategy trade summary 
    summary_rows = []

    best_trades = []
    bestProfit = -np.inf
    for i, trade_set in enumerate(trades):
        # transform list of dictionaries to dataframe
        df = pd.DataFrame(trade_set)

        if df.empty:
            row = {
                'Strategy' : i,
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
                'Strategy' : i,
                'Total Profit (%)': profit,
                'Median % Change': df['pct_change'].median(),
                'Max % Change': df['pct_change'].max(),
                'Number of Trades': len(df),
                'Win Rate (%)': df['win'].mean() * 100,
                'Avg Holding Time' : df['time-transpired'].mean().round('s')
            }

        # seperatly store the best strategy based on profit
        if profit > bestProfit:
            bestProfit = profit
            best_trades = trade_set

        # store the row regardless
        summary_rows.append(row)

    # turn it into a dataframe
    summary = pd.DataFrame(summary_rows)
    best_tradesdf = pd.DataFrame(best_trades)

    # save the data in csv
    mvData.to_csv("Data/FoundStrats/TopStrats15M50k.csv", index=False)
    best_tradesdf.to_csv("Data/Trades2Vis/trades15M50k.csv", index=False)

    # Optionally round
    summary = summary.round(2)

    # Display
    print(summary)

    end = time.time()
    print("Time spend:", end-start)
