import pandas as pd

def exitTrade(t, entry_value):
    """
    This function exits the trade by creating a trade description to be added to a list of trades. 
    """
    # compute the trade movement in percentages
    pct_change = (t['ex_p'] - t['en_p']) / t['en_p']

    # compute profit based on value change minus transaction cost
    profit = (pct_change * entry_value) - (entry_value * 0.004 + (1 + pct_change) * entry_value * 0.004)

    TradeDescription = {
        'entry-price': t['en_p'],
        'exit-price': t['ex_p'],
        'pct_change': pct_change * 100,
        'profit': profit,
        'entry-time': t['en_t'],
        'exit-time': t['ex_t'],
        'time-transpired': t['ex_t'] - t['en_t']
        }
    
    return TradeDescription, profit

def TradeSim(program, priceD):
    """"
    Used to simulate trades for a given program and data.
    """
    in_trade = False
    entry_price = 0
    entry_time = None
    trades = []
    funds = 100

    for i, row in priceD.iterrows():

        # evaluate the rule for each row
        try:
            entry_rule, exit_rule, trade_size, sloss = program(priceD, i)

        except Exception as e:
            print(f"Error evaluating rule: {e}")
            return -1000

        # if it's not in a trade and a valid entry is indicated then enter the trade
        if not in_trade and entry_rule:
            in_trade = True
            entry_price = row['close']
            entry_time = row['time']

            # compute the trade size and respective sloss
            entry_value = trade_size * funds

            # assign SLossPrice sloss, if sloss is not 0 then assign it the actual stop loss price
            if SLossPrice := sloss:
                SLossPrice = entry_price - entry_price * (sloss/100)

        # if it's in a trade and a valid exit is indicated then finish the trade
        elif in_trade:

            # if the stop loss condition is met
            if row['low'] <= SLossPrice:

                # store stop loss price as exist price
                exit_time = row['time']

                # store the info of the trade
                trade_info = {'en_t': entry_time, 'en_p': entry_price, 'ex_p': SLossPrice, 'ex_t': exit_time}

                # exit the trade and store its information and resulting profit
                tradeFinal, profit = exitTrade(trade_info, entry_value)

                # store the trade information
                trades.append(tradeFinal)

                # update the funds
                funds += profit

                # allow new trades
                in_trade = False

            # if stop loss hasn't been cross and exit conditions are met
            elif exit_rule:

                # store trade exit
                exit_price = row['close']
                exit_time = row['time']

                # store the info of the trade
                trade_info = {'en_t': entry_time, 'en_p':entry_price, 'ex_p': exit_price, 'ex_t': exit_time}

                # exit the trade and store its information and resulting profit
                tradeFinal, profit = exitTrade(trade_info, entry_value)

                # store the trade information
                trades.append(tradeFinal)

                # update the funds
                funds += profit

                # allow new trades
                in_trade = False

    return pd.DataFrame(trades)