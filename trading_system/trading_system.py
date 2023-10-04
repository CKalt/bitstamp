"""
Module: trading_system.py
Description: Contains the TradingSystem class responsible for starting the trading system and executing trades based on provided data and decisions from the trading engine.
"""

from data_feeder import DataFeeder
from trading_engine import TradingEngine


class TradingSystem:
    """
    A class used to represent the Trading System.

    ...

    Methods
    -------
    __init__(mode, file_paths)
        Initializes the Trading System with the mode, file paths, and other necessary components.
    start()
        Starts the trading system, executes trades, and prints the profit for each executed trade.
    """

    def __init__(self, mode, file_paths, parameters):
        """
        Initializes the Trading System with the mode, file paths, and other necessary components.
        :param mode: The mode of the trading system (realtime or playback).
        :param file_paths: The file paths for trade data in playback mode.
        :param parameters: The trading parameters.
        """
        self.mode = mode
        self.file_paths = file_paths
        self.data_feeder = DataFeeder()
        self.trading_engine = TradingEngine()
        self.balances = parameters["initial_balances"]

    def start(self):
        """
        Starts the trading system, executes trades, and prints the profit for each executed trade.
        :return: Final balances and total profit.
        """
        total_profit = 0
        data_streams = self.data_feeder.get_data(self.mode, self.file_paths)
        while True:
            try:
                data = {pair: next(stream) for pair, stream in data_streams.items()}
                new_balances, profit = self.trading_engine.make_decision(data, self.balances, self.parameters)
                self.balances.update(new_balances)
                total_profit += profit
                if profit > 0:
                    print(f"Executed trade with profit: {profit}")
            except StopIteration:
                break
        return self.balances, total_profit
