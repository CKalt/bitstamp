"""
Module: trading_engine.py
Description: Contains the TradingEngine class responsible for making trading decisions based on provided data and parameters.
"""

import json
from trade_executor import TradeExecutor


class TradingEngine:
    """
    A class used to represent the Trading Engine.

    ...

    Methods
    -------
    make_decision(data, balances, parameters)
        Makes trading decisions based on provided data and parameters.
    extract_prices(data)
        Extracts and returns prices from the provided data.
    calculate_arbitrage_opportunity(prices, parameters)
        Calculates and returns the arbitrage opportunity based on provided prices and parameters.
    """

    def make_decision(self, data, balances, parameters):
        """
        Make trading decisions based on provided data and parameters.
        :param data: The trade data.
        :param balances: The balances of different currencies.
        :param parameters: The trading parameters.
        :return: Updated balances and profit.
        """
        prices = self.extract_prices(data)
        cycle, opportunity = self.calculate_arbitrage_opportunity(prices, parameters)

        # Checking if the opportunity is greater than the threshold to execute the trade
        if opportunity > parameters['arbitrage_opportunity_threshold']:
            trade_executor = TradeExecutor()
            new_balances, profit = trade_executor.execute_arbitrage_trade(data, cycle, balances, parameters['fee'])
            balances.update(new_balances)
            return balances, profit
        return balances, 0

    def extract_prices(self, data):
        """
        Extract prices from the data.
        :param data: The trade data.
        :return: A dictionary of prices.
        """
        return {
            'bchbtc': float(data['bchbtc'][1]),
            'bchusd': float(data['bchusd'][1]),
            'btcusd': float(data['btcusd'][1])
        }

    def calculate_arbitrage_opportunity(self, prices, parameters):
        """
        Calculates and returns the arbitrage opportunity based on provided prices and parameters.
        :param prices: The prices of different currency pairs.
        :param parameters: The trading parameters.
        :return: The cycle and arbitrage opportunity.
        """
        btc_balance = 1
        effective_btc_1 = btc_balance / prices['bchbtc'] * prices['bchusd'] / prices['btcusd']
        effective_btc_2 = btc_balance * prices['bchbtc'] * prices['bchusd'] / prices['btcusd']
        opportunity1 = abs(effective_btc_1 - btc_balance)
        opportunity2 = abs(effective_btc_2 - btc_balance)
        return (1, opportunity1) if opportunity1 > opportunity2 else (2, opportunity2)
