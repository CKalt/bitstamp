import logging

class TradingEngine:
    def make_decision(self, data, balances, parameters):
        try:
            prices = self.extract_prices(data)
            opportunities = self.calculate_opportunities(prices, balances)
            print(f"Data: {data}, Opportunity1: {opportunities[0]}, Opportunity2: {opportunities[1]}")

            cycle = self.determine_cycle(opportunities, parameters)
            if cycle:
                return cycle, opportunities[cycle - 1]
            return None, None
        except Exception as e:
            logging.error(f"Error making decision: {e}")
            print(f"Error making decision: {e}")

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
        opportunity1 = (btc_balance * prices['bchbtc'] * prices['bchusd']) / prices['btcusd'] - btc_balance
        opportunity2 = (btc_balance / prices['btcusd'] / prices['bchusd']) * prices['bchbtc'] - btc_balance
        return opportunity1, opportunity2

    def determine_cycle(self, opportunities, parameters):
        """Determine the cycle based on opportunities and parameters."""
        if opportunities[0] > parameters['arbitrage_opportunity_threshold']:
            return 1
        elif opportunities[1] > parameters['arbitrage_opportunity_threshold']:
            return 2
        return None
