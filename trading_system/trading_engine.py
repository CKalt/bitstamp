
"""
Module: trading_engine.py
Description: Contains the TradingEngine class responsible for making trading decisions based on provided data and parameters.
"""

import json


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
    calculate_opportunities(prices, balances)
        Calculates and returns trading opportunities based on prices and balances.
    determine_cycle(opportunities, parameters)
        Determines and returns the trading cycle based on opportunities and parameters.
    """

    def make_decision(self, data, balances, parameters):
        """
        Makes trading decisions based on provided data and parameters.

        Parameters:
            data (dict): A dictionary containing price data.
            balances (dict): A dictionary containing balance data.
            parameters (dict): A dictionary containing various parameters.

        Returns:
            tuple: A tuple containing the cycle and opportunity data.
        """
        try:
            prices = self.extract_prices(data)
            opportunities = self.calculate_opportunities(prices, balances)
            print(
                f"Data: {data}, Opportunity1: {opportunities[0]}, Opportunity2: {opportunities[1]}")

            cycle = self.determine_cycle(opportunities, parameters)
            if cycle:
                return cycle, opportunities[cycle - 1]
            return None, None
        except Exception as e:
            error_info = {"level": "error", "message": str(e)}
            with open(parameters['output_files']['log_file'], 'a') as log_file:
                log_file.write(json.dumps(error_info) + '\n')

    def extract_prices(self, data):
        """Extract prices from the data."""
        return {
            'bchbtc': data['bchbtc'].get('price', 1),
            'bchusd': data['bchusd'].get('price', 1),
            'btcusd': data['btcusd'].get('price', 1)
        }

    def calculate_opportunities(self, prices, balances):
        """Calculate the opportunities based on prices and balances."""
        btc_balance = balances['BTC']
        opportunity1 = (
            btc_balance * prices['bchbtc'] * prices['bchusd']) / prices['btcusd'] - btc_balance
        opportunity2 = (
            btc_balance / prices['btcusd'] / prices['bchusd']) * prices['bchbtc'] - btc_balance
        return opportunity1, opportunity2

    def determine_cycle(self, opportunities, parameters):
        """Determine the cycle based on opportunities and parameters."""
        if opportunities[0] > parameters['arbitrage_opportunity_threshold']:
            return 1
        elif opportunities[1] > parameters['arbitrage_opportunity_threshold']:
            return 2
        return None
