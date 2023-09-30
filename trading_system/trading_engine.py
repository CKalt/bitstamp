import logging

class TradingEngine:
    def make_decision(self, data, balances, parameters):
        try:
            bchbtc_price = data['bchbtc'].get('price', 1)
            bchusd_price = data['bchusd'].get('price', 1)
            btcusd_price = data['btcusd'].get('price', 1)
            btc_balance = balances['BTC']

            opportunity1 = (btc_balance * bchbtc_price * bchusd_price) / btcusd_price - btc_balance
            opportunity2 = (btc_balance / btcusd_price / bchusd_price) * bchbtc_price - btc_balance

            print(f"Data: {data}, Opportunity1: {opportunity1}, Opportunity2: {opportunity2}")

            if opportunity1 > parameters['arbitrage_opportunity_threshold']:
                return 'cycle1', opportunity1
            elif opportunity2 > parameters['arbitrage_opportunity_threshold']:
                return 'cycle2', opportunity2
            return None, None
        except Exception as e:
            logging.error(f"Error making decision: {e}")
            print(f"Error making decision: {e}")
