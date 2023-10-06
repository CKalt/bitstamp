# bitstamp
src/triangular.py 

In the provided code, the triangular arbitrage strategy is being used to explore profit opportunities by trading across three pairs: BTC to BCH (bchbtc), BCH to USD (bchusd), and USD back to BTC (btcusd). For each of these trades, there's a transaction fee applied. Let's break down the calculations with respect to fees:

    Trading BTC for BCH:
        Initial amount: 1 BTC.
        Conversion formula: amount_in_bch = (1.0 / price_in_bchbtc) * (1 - transaction_fee)
        This essentially says that you take the inverse of the bchbtc price to get the amount in BCH you would receive for 1 BTC. You then subtract the transaction fee from this amount.

    Trading BCH for USD:
        Conversion formula: amount_in_usd = amount_in_bch * price_in_bchusd * (1 - transaction_fee)
        Here, you're multiplying the amount of BCH you have by the bchusd price to get the equivalent amount in USD. You then account for the transaction fee by subtracting it.

    Trading USD back to BTC:
        Conversion formula: final_btc_amount = amount_in_usd / price_in_btcusd * (1 - transaction_fee)
        You divide the amount in USD by the btcusd price to convert back to BTC, then subtract the transaction fee.

Finally, the profit or loss is calculated by subtracting the original amount (1 BTC) from the final_btc_amount.

Profit/Loss Calculation:

    profit_or_loss = final_btc_amount - 1.0

If profit_or_loss is positive and above the defined profit_threshold, then it's a profitable arbitrage opportunity.

Regarding the transaction fees:

    The transaction_fee is taken as a percentage (in the code, it's set to 0.002, which is 0.2%).
    For each of the trades, this fee is applied, which means the actual amount being traded gets reduced by this fee.

In real-world scenarios, the fee structures might be more complex, with variable fees based on trade volumes, specific platforms, or other factors. However, this code assumes a constant fee for simplicity. If you're planning to implement this in a real-world scenario, it's essential to account for these nuances.
