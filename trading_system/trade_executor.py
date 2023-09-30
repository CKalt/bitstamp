"""
Module: trade_executor.py
Description: Contains the TradeExecutor class responsible for executing trades based on the provided trade data and updating balances.
"""

import json


class TradeExecutor:
    def execute_arbitrage_trade(self, trade_data, cycle, balances, fee):
        initial_balances = balances.copy()
        if cycle == 1:
            # Execute trades for Cycle 1: USD -> BTC -> BCH -> USD
            balances = self.execute_trade(balances, ("USD", "BTC"), trade_data['btcusd'], balances['USD'], fee)
            balances = self.execute_trade(balances, ("BTC", "BCH"), trade_data['bchbtc'], balances['BTC'], fee)
            balances = self.execute_trade(balances, ("BCH", "USD"), trade_data['bchusd'], balances['BCH'], fee)
        elif cycle == 2:
            # Execute trades for Cycle 2: USD -> BCH -> BTC -> USD
            balances = self.execute_trade(balances, ("USD", "BCH"), trade_data['bchusd'], balances['USD'], fee)
            balances = self.execute_trade(balances, ("BCH", "BTC"), 1 / trade_data['bchbtc'], balances['BCH'], fee)
            balances = self.execute_trade(balances, ("BTC", "USD"), 1 / trade_data['btcusd'], balances['BTC'], fee)
        profit = balances['USD'] - initial_balances['USD']
        return balances, profit

    """
    A class used to represent the Trade Executor.

    ...

    Methods
    -------
    execute_trade(trade, balances, parameters)
        Executes the trade based on the provided trade data and updates the balances.
    calculate_fee(opportunity, parameters)
        Calculates the trading fee based on the opportunity and fee parameter.
    calculate_required_btc(balances, parameters)
        Calculates the required BTC based on the balances and risk measure parameter.
    execute_cycle1_trade(balances, required_btc)
        Executes a cycle1 trade and updates the balances.
    execute_cycle2_trade(balances, required_btc)
        Executes a cycle2 trade and updates the balances.
    """

    def execute_trade(self, trade, balances, parameters):
        """
        Executes the trade based on the provided trade data and updates the balances.

        Parameters:
            trade (tuple): A tuple containing the cycle and opportunity data.
            balances (dict): A dictionary containing the balance data.
            parameters (dict): A dictionary containing various parameters including fee and risk measure.

        Returns:
            None
        """
        try:
            trade_info = {"message": f"Executing trade: {trade}"}
            with open(parameters['output_files']['log_file'], 'a') as log_file:
                log_file.write(json.dumps(trade_info) + '\n')
            cycle, opportunity = trade
            fee = self.calculate_fee(opportunity, parameters)
            required_btc = self.calculate_required_btc(balances, parameters)

            if cycle == 'cycle1':
                self.execute_cycle1_trade(balances, required_btc)
            elif cycle == 'cycle2':
                self.execute_cycle2_trade(balances, required_btc)

            profit = opportunity - fee
        except Exception as e:
            error_info = {"level": "error", "message": str(e)}
            with open(parameters['output_files']['log_file'], 'a') as log_file:
                log_file.write(json.dumps(error_info) + '\n')

    def calculate_fee(self, opportunity, parameters):
        """Calculate the fee based on the opportunity and parameters."""
        return opportunity * parameters['fee']

    def calculate_required_btc(self, balances, parameters):
        """Calculate the required BTC based on the balances and parameters."""
        return parameters['risk_measure'] * balances['BTC']

    def execute_cycle1_trade(self, balances, required_btc):
        """Execute a cycle1 trade."""
        if balances['BTC'] >= required_btc:
            balances['BTC'] -= required_btc
            balances['BCH'] += required_btc * balances['BCHBTC']
            balances['USD'] += balances['BCH'] * balances['BCHUSD']
            balances['BTC'] += balances['USD'] / balances['BTCUSD']

    def execute_cycle2_trade(self, balances, required_btc):
        """Execute a cycle2 trade."""
        if balances['BTC'] >= required_btc:
            balances['BTC'] -= required_btc
            balances['USD'] += required_btc * balances['BTCUSD']
            balances['BCH'] += balances['USD'] / balances['BCHUSD']
            balances['BTC'] += balances['BCH'] * balances['BCHBTC']
