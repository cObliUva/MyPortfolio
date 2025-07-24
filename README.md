## Trading Strategy Generator

Method: Use LOTlib3

Input: Price data

Output: Optimal Trade Strategies

Likelihood, computed by 


### OLD Step Plan
1. Create price plot that includes RSI, and trades alongside moving averages/bollinger bands. - DONE
2. Create a function that finds trade entries based on RSI divergence. - DONE
3. Create a function that finds trade exits based on RSI divergence. - DONE (Not a succesfull strategy)
4. Create a function that computes trade strategy success based on win-rate, trade frequency, and profits. Profits * win-rate * frequency?? - DONEISH
5. Create TopStrat.py containing working version with the easiest grammar and strategy. - DONEISH
6. Implement more complex variables alongside a grammar that enables advanced strategies. - WORKING
7. Get good strategy that makes sense.
8. Combine time-frames to make more holistically informed decisions.

### NEW Step Plan To Trading-Bot
1. First complete the 1 Day - Week trading strategy prototype, and backtest on other data.
    -  Create more advanced features that are low in train-time complexity (pivot-high/low detection, and price-range logic (kda), RSI-divergence)
2. Create a second model that creates swing-trade strategies on 1 day - 1 hour time-frame. (Create a new file for this, but use a lot of the same code as a framework).
3. Create a scalping strategy that works with 1 hours to 5 min data.
    - Use the strategy and create a tradingbot that would use API of a crypto exchange to actually make trades.
    - Create trading data dashboard (i.e., already created plots + trade data, but also store trade data in csv and create good tables of performance).
    - First do paper-trading with the bot on live data.

### Symbolic Regression Implementation
Data = Price and indicator per candle

Normally seperated by case or trial, different examples of something to learn from.

Possible split up the data into random samples that a trading strategy should be made for. Possible to make data objects that showcase different time-frame combinations to determine exact trade price for larger moments.

Attempt to find a specific trading strategy that works best for certain time-frames. 
1. One for year-month basis long-term buy and sell points.
2. Another for week-day basis, for week on week voltatility.
3. A third for day-hour basis, to make short term trades.

Include as many samples as possible to determine consistent performance for a strategy that maximizes profit for each sample.



#### Problem 1

The algorithm created by LOTlib3 simply creates a single lambda expression that calls various functions that produces answer which minimize a loss function. Normally there are correct answers that are attempted to be reproduced by a generating expression.

In this case, we're maximizing profit by following a rule which determines good buy and sell moments.

Use if ROI > 0:
        likelihood = ROI * winrate
    else: ROI

Comes from (Menoita & Silva, 2025).



**Solution 1:** Create rules

