
import json
from data_feeder import DataFeeder
from trading_engine import TradingEngine


class TradingSystem:
    def __init__(self, mode, file_paths):
        self.mode = mode
        self.file_paths = file_paths
        self.data_feeder = DataFeeder()
        self.trading_engine = TradingEngine()
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
            try:
                data = {pair: next(stream) for pair, stream in data_streams.items()}
                data_info = {"message": f"Processing data: {data}"}
                with open(self.parameters['output_files']['log_file'], 'a') as log_file:
                    log_file.write(json.dumps(data_info) + '\n')

                # Updated part: Using the new make_decision method and handling its return values.
                new_balances, profit = self.trading_engine.make_decision(data, self.balances, self.parameters)
                self.balances.update(new_balances)
                if profit > 0:
                    trade_info = {"message": f"Executed trade with profit: {profit}"}
                    with open(self.parameters['output_files']['log_file'], 'a') as log_file:
                        log_file.write(json.dumps(trade_info) + '\n')
            except StopIteration:
                break  # Exit the loop when no more data is available
