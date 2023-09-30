
import json
from data_feeder import DataFeeder
from trading_engine import TradingEngine
from trade_executor import TradeExecutor

class TradingSystem:
    def __init__(self, mode, file_paths):
        self.mode = mode
        self.file_paths = file_paths
        self.data_feeder = DataFeeder()
        self.trading_engine = TradingEngine()
        self.trade_executor = TradeExecutor()
        self.balances = {}  
        self.parameters = self.load_parameters()
    
    def load_parameters(self):
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
        self.balances = parameters["initial_balances"]
        return parameters
    
    def start(self):
        data_streams = self.data_feeder.get_data(self.mode, self.file_paths)
        while True:
            data = {pair: next(stream) for pair, stream in data_streams.items()}
            decision, opportunity = self.trading_engine.make_decision(data, self.balances, self.parameters)
            if decision:
                self.trade_executor.execute_trade((decision, opportunity), self.balances, self.parameters)
