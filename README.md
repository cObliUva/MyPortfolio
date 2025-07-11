## Trading Strategy Generator

Method: Use LOTlib3

Input: Price data

Output: Optimal Trade Strategies

Likelihood, computed by 


### Step Plan
1. Create price plot that includes RSI, and trades alongside moving averages/bollinger bands. - DONE
2. Create a function that finds trade entries based on RSI divergence. - DONE
3. Create a function that finds trade exits based on RSI divergence. - DONE (Not a succesfull strategy)
4. Create a function that computes trade strategy success based on win-rate, trade frequency, and profits. Profits * win-rate * frequency?? - DONE

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

