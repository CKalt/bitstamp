
import unittest
from trading_engine import TradingEngine

class TestTradingEngine(unittest.TestCase):
    def setUp(self):
        self.trading_engine = TradingEngine()

    def test_extract_prices(self):
        data = {
            'bchbtc': {'price': 0.05},
            'bchusd': {'price': 500},
            'btcusd': {'price': 10000}
        }  # Example data
        result = self.trading_engine.extract_prices(data)
        self.assertEqual(result, {'bchbtc': 0.05, 'bchusd': 500, 'btcusd': 10000})

if __name__ == '__main__':
    unittest.main()
