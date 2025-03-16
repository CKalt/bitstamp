###############################################################################
# File Path: src/strategies/ramm_strategy.py
###############################################################################
# CHANGES:
#   1) In the new RAMMStrategy class, whenever we set position=1 from <=0,
#      we now do partial_buy_3x_90pct(...) from the base class.
#   2) We keep all original logic, comments, and code except for that change.
###############################################################################

from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    calculate_rsi,
    calculate_market_conditions
)
from strategies.base_strategy import BaseStrategy

def calculate_ramm_signals(df,
                           ma_short=10, ma_long=50,
                           rsi_period=14, rsi_ob=70, rsi_os=30,
                           regime_lookback=20):
    """
    Generate RAMM strategy signals combining MA Crossover and RSI based on market regime
    (unchanged from original).
    """
    df = ensure_datetime_index(df)

    # Calculate market regime
    df_regime = calculate_market_conditions(df.copy(), regime_lookback)
    df['regime'] = df_regime['regime']

    # Calculate MA signals
    df = add_moving_averages(df, ma_short, ma_long)
    df['MA_Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'MA_Signal'] = 1
    df.loc[df['Short_MA'] < df['Long_MA'], 'MA_Signal'] = -1

    # Calculate RSI signals
    df = calculate_rsi(df, rsi_period)
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < rsi_os, 'RSI_Signal'] = 1
    df.loc[df['RSI'] > rsi_ob, 'RSI_Signal'] = -1

    # Combine them
    df['RAMM_Signal'] = 0

    # Trending regime => use MA
    df.loc[df['regime'] == 1, 'RAMM_Signal'] = df.loc[df['regime'] == 1, 'MA_Signal']

    # Mean-reverting => use RSI
    df.loc[df['regime'] == -1, 'RAMM_Signal'] = df.loc[df['regime'] == -1, 'RSI_Signal']

    # Mixed => only trade if both agree
    mixed_mask = df['regime'] == 0
    df.loc[mixed_mask & (df['MA_Signal'] == 1) & (df['RSI_Signal'] == 1), 'RAMM_Signal'] = 1
    df.loc[mixed_mask & (df['MA_Signal'] == -1) & (df['RSI_Signal'] == -1), 'RAMM_Signal'] = -1

    return df

###############################################################################
# NEW: Live-trading RAMMStrategy class that inherits from BaseStrategy
###############################################################################
class RAMMStrategy(BaseStrategy):
    """
    RAMM strategy for live trading. Now ensures any time we go long, we do
    3 partial buys of 90% each from the base strategy.
    """

    def __init__(
        self,
        data_manager,
        logger,
        live_trading=False,
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0,
        ma_short=10,
        ma_long=50,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        regime_lookback=20,
        **kwargs
    ):
        super().__init__(
            data_manager=data_manager,
            logger=logger,
            live_trading=live_trading,
            max_trades_per_day=max_trades_per_day,
            initial_position=initial_position,
            initial_balance_btc=initial_balance_btc,
            initial_balance_usd=initial_balance_usd,
            **kwargs
        )
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.regime_lookback = regime_lookback

        self.symbol = "btcusd"
        self.strategy_thread = None
        self.last_signal_time = None

        # Additional tracking
        self.bar_size = '1H'
        self.theoretical_trade = None
        self.position_size = 0.0
        self.position_cost_basis = 0.0
        self.df_ramm = None
        self.logger.info("RAMMStrategy initialized with partial buy logic for going long.")

    def start(self):
        import threading
        self.running = True
        self.strategy_thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("RAMM strategy loop started.")

    def stop(self):
        self.running = False
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join()
        self.logger.info("RAMM strategy loop stopped.")

    def run_strategy_loop(self):
        import time
        from indicators.technical_indicators import ensure_datetime_index
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample(self.bar_size).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'trades': 'sum',
                        'timestamp': 'last',
                        'source': 'last'
                    }).dropna()

                    # Enough data?
                    lookback = max(self.ma_long, self.rsi_period, self.regime_lookback)
                    if len(df_resampled) >= lookback:
                        self.df_ramm = calculate_ramm_signals(
                            df_resampled.copy(),
                            ma_short=self.ma_short,
                            ma_long=self.ma_long,
                            rsi_period=self.rsi_period,
                            rsi_ob=self.rsi_overbought,
                            rsi_os=self.rsi_oversold,
                            regime_lookback=self.regime_lookback
                        )
                        latest_signal = self.df_ramm.iloc[-1]['RAMM_Signal']
                        signal_time = self.df_ramm.index[-1]
                        current_price = self.df_ramm.iloc[-1]['close']
                        self.check_for_signals(latest_signal, current_price, signal_time)
                except Exception as e:
                    self.logger.error(f"Error in RAMM strategy loop: {e}")
            else:
                self.logger.debug("RAMMStrategy: No data loaded yet.")
            time.sleep(60)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        If signal=1 and position <=0 => do partial 3-step buy
        If signal=-1 and position >=0 => do single sell
        """
        if self.last_signal_time == signal_time:
            return
        if not self.check_daily_limit():
            return

        if latest_signal == 1 and self.position <= 0:
            self.logger.info(f"RAMM: Buy signal triggered at {current_price}")
            self.position = 1
            from datetime import datetime
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.partial_buy_3x_90pct(
                price=current_price,
                timestamp_str=now_str,
                signal_time=signal_time
            )
            self.last_signal_time = signal_time

        elif latest_signal == -1 and self.position >= 0:
            self.logger.info(f"RAMM: Sell signal triggered at {current_price}")
            trade_btc = round(self.balance_btc, 8)
            from datetime import datetime
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.execute_trade(
                trade_type="sell",
                price=current_price,
                timestamp=now_str,
                signal_time=signal_time,
                trade_btc=trade_btc,
                is_partial=False
            )
            self.trade_count_today += 1
            self.position = -1
            self.last_signal_time = signal_time
