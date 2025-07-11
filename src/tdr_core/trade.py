# src/tdr_core/trade.py

from datetime import datetime

###############################################################################
class Trade:
    """
    Represents a single trade, whether historical or live.
    """
    def __init__(
        self,
        trade_type,
        symbol,
        amount,
        price,
        timestamp,
        reason,
        data_source,
        signal_timestamp,
        live_trading=False,
        order_result=None,
        trade_group_id=None,
        multi_part_sequence=None,
        multi_part_total=None
    ):
        self.type = trade_type
        self.symbol = symbol
        self.amount = amount
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        self.data_source = data_source
        self.signal_timestamp = signal_timestamp
        self.live_trading = live_trading
        self.order_result = order_result
        # Multi-part trade tracking
        self.trade_group_id = trade_group_id  # Unique ID for the entire multi-part trade
        self.multi_part_sequence = multi_part_sequence  # Part number (1, 2, 3, etc.)
        self.multi_part_total = multi_part_total  # Total number of parts

    def to_dict(self):
        """
        Convert the Trade object to a dictionary for logging/JSON.
        """
        trade_info = {
            'type': self.type,
            'symbol': self.symbol,
            'amount': self.amount,
            'price': self.price,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal_timestamp': self.signal_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': self.data_source,
            'live_trading': self.live_trading,
            'reason': self.reason
        }
        if self.order_result:
            trade_info['order_result'] = self.order_result
        # Add multi-part trade information if present
        if self.trade_group_id is not None:
            trade_info['trade_group_id'] = self.trade_group_id
        if self.multi_part_sequence is not None:
            trade_info['multi_part_sequence'] = self.multi_part_sequence
        if self.multi_part_total is not None:
            trade_info['multi_part_total'] = self.multi_part_total
        return trade_info
