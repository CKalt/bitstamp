"""
Auto-Resume Stability Fixes for Live Branch

This module contains improved functions for auto-resume functionality
to address the following issues:
1. Wrong amounts saved (USD for LONG positions)
2. Incorrect entry price calculations for multi-part trades
3. Lack of proper validation of resume data
4. Silent failures during resume
"""

import json
import os
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


class ResumeValidator:
    """Validates resume data for consistency and correctness"""
    
    @staticmethod
    def validate_resume_data(resume_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate resume data structure and values.
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['position', 'amount', 'unit', 'entry_price', 'command']
        for field in required_fields:
            if field not in resume_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
            
        # Validate position type
        position = resume_data.get('position', '').upper()
        if position not in ['LONG', 'SHORT']:
            errors.append(f"Invalid position type: {position}")
            
        # Validate unit matches position
        unit = resume_data.get('unit', '').lower()
        if position == 'LONG' and unit != 'btc':
            errors.append(f"LONG position must use BTC unit, not {unit}")
        elif position == 'SHORT' and unit != 'usd':
            errors.append(f"SHORT position must use USD unit, not {unit}")
            
        # Validate amounts
        amount = resume_data.get('amount', 0)
        if amount <= 0:
            errors.append(f"Invalid amount: {amount}")
            
        # Validate entry price
        entry_price = resume_data.get('entry_price', 0)
        if entry_price <= 0:
            errors.append(f"Invalid entry price: {entry_price}")
            
        # Validate command format
        command = resume_data.get('command', '')
        if not command.startswith('resume_auto_trade'):
            errors.append(f"Invalid command format: {command}")
            
        return len(errors) == 0, errors


def calculate_correct_entry_price(trades: List[Dict], position_type: str) -> Tuple[float, List[Dict]]:
    """
    Calculate the correct entry price from trades based on position type.
    
    For LONG positions: Weight-average all BUY prices
    For SHORT positions: Use the entry SELL price (first sell of the position)
    
    Returns: (entry_price, position_trades)
    """
    if not trades:
        return 0.0, []
        
    # Find all trades for current position
    position_trades = []
    current_position = None
    
    # Work backwards to find current position trades
    for trade in reversed(trades):
        trade_type = trade.get('type', '').lower()
        
        if not current_position:
            # First trade determines position type
            current_position = 'LONG' if trade_type == 'buy' else 'SHORT'
            position_trades.append(trade)
        elif (current_position == 'LONG' and trade_type == 'buy') or \
             (current_position == 'SHORT' and trade_type == 'sell'):
            # Same direction trade, part of position
            position_trades.append(trade)
        else:
            # Opposite direction trade, previous position closed
            break
    
    # Reverse to get chronological order
    position_trades.reverse()
    
    if not position_trades:
        return 0.0, []
        
    # Validate position type matches
    if position_type.upper() != current_position:
        logger.warning(f"Position type mismatch: expected {position_type}, found {current_position}")
        return 0.0, []
    
    # Calculate entry price based on position type
    if current_position == 'LONG':
        # For LONG: weight-average all BUY prices
        total_btc = sum(float(t.get('amount', 0)) for t in position_trades)
        total_cost = sum(float(t.get('amount', 0)) * float(t.get('price', 0)) 
                        for t in position_trades)
        entry_price = total_cost / total_btc if total_btc > 0 else 0
        logger.info(f"Calculated LONG entry from {len(position_trades)} trades: "
                   f"${entry_price:.2f} (total BTC: {total_btc:.8f})")
    else:  # SHORT
        # For SHORT: use the first SELL price (entry price)
        entry_price = float(position_trades[0].get('price', 0))
        total_usd = sum(float(t.get('amount', 0)) * float(t.get('price', 0)) 
                       for t in position_trades)
        logger.info(f"Using SHORT entry price: ${entry_price:.2f} "
                   f"(total USD: ${total_usd:.2f})")
    
    return entry_price, position_trades


def create_reliable_resume_data(strategy) -> Dict:
    """
    Create reliable resume data with proper validation and error handling.
    
    Args:
        strategy: The trading strategy instance (MACrossoverStrategy)
        
    Returns:
        Dict containing validated resume data
    """
    try:
        # Get current price
        current_price = strategy.data_manager.get_current_price(strategy.symbol) or 0.0
        
        # Load trades for entry price calculation
        trades = []
        if os.path.exists(strategy.trade_log_file):
            with open(strategy.trade_log_file, 'r') as f:
                trades = json.load(f)
        
        # Determine position type and validate
        if strategy.position == 1:  # LONG
            position_type = 'long'
            amount = strategy.balance_btc
            unit = 'btc'
            
            # Validate BTC balance
            if amount <= 0:
                raise ValueError(f"Invalid BTC balance for LONG position: {amount}")
                
        elif strategy.position == -1:  # SHORT
            position_type = 'short'
            amount = strategy.balance_usd
            unit = 'usd'
            
            # Validate USD balance
            if amount <= 0:
                raise ValueError(f"Invalid USD balance for SHORT position: {amount}")
                
        else:
            raise ValueError(f"Invalid position state: {strategy.position}")
        
        # Calculate correct entry price from trades
        entry_price, position_trades = calculate_correct_entry_price(trades, position_type)
        
        # Fallback entry price calculation if needed
        if entry_price <= 0:
            logger.warning("Could not calculate entry price from trades, using fallback")
            if strategy.position_size != 0:
                entry_price = abs(strategy.position_cost_basis / strategy.position_size)
            else:
                entry_price = strategy.last_trade_price or current_price
        
        # Calculate unrealized P&L
        if position_type == 'long':
            unrealized_pnl = amount * (current_price - entry_price)
        else:  # short
            # For short: profit when price goes down
            btc_equivalent = amount / entry_price
            unrealized_pnl = btc_equivalent * (entry_price - current_price)
        
        # Create resume data
        resume_data = {
            'timestamp': datetime.now().isoformat(),
            'position': position_type.upper(),
            'amount': round(amount, 8),
            'unit': unit,
            'entry_price': round(entry_price, 2),
            'current_price': round(current_price, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'command': f"resume_auto_trade {amount:.8f}{unit} {position_type} {entry_price:.0f}",
            'strategy': {
                'type': 'MACrossoverStrategy',
                'short_window': strategy.short_window,
                'long_window': strategy.long_window,
                'max_trades_per_day': strategy.max_trades_per_day
            },
            'balances': {
                'btc': round(strategy.balance_btc, 8),
                'usd': round(strategy.balance_usd, 2)
            },
            'position_tracking': {
                'position': strategy.position,
                'position_size': round(strategy.position_size, 8),
                'position_cost_basis': round(strategy.position_cost_basis, 2)
            },
            'trades_executed': strategy.trades_executed,
            'last_trade_time': strategy.last_trade_time.isoformat() if strategy.last_trade_time else None,
            'validation': {
                'version': '2.0',
                'validated': True,
                'trade_count': len(position_trades)
            }
        }
        
        # Add trade references if available
        if position_trades:
            resume_data['trade_references'] = [
                {
                    'timestamp': t['timestamp'],
                    'type': t['type'],
                    'amount': t['amount'],
                    'price': t['price'],
                    'trade_group_id': t.get('trade_group_id', 'unknown')
                }
                for t in position_trades[-5:]  # Last 5 trades only
            ]
        
        # Validate the resume data before returning
        is_valid, errors = ResumeValidator.validate_resume_data(resume_data)
        if not is_valid:
            raise ValueError(f"Invalid resume data: {', '.join(errors)}")
            
        return resume_data
        
    except Exception as e:
        logger.error(f"Failed to create resume data: {e}", exc_info=True)
        raise


def save_resume_state_with_validation(strategy) -> bool:
    """
    Save resume state with proper validation and error handling.
    
    Args:
        strategy: The trading strategy instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create validated resume data
        resume_data = create_reliable_resume_data(strategy)
        
        # Find resume file path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resume_file = os.path.join(project_root, 'resume-auto-trade.json')
        
        # Create backup of existing file
        if os.path.exists(resume_file):
            backup_file = f"{resume_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(resume_file, 'r') as f:
                backup_data = json.load(f)
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            logger.info(f"Created backup: {backup_file}")
        
        # Save new resume data
        with open(resume_file, 'w') as f:
            json.dump(resume_data, f, indent=2)
            
        logger.info(f"✅ Resume state saved successfully: {resume_data['position']} "
                   f"{resume_data['amount']:.8f} {resume_data['unit'].upper()} "
                   f"@ ${resume_data['entry_price']:.2f}")
        
        # Verify the file was written correctly
        with open(resume_file, 'r') as f:
            verified_data = json.load(f)
            
        if verified_data != resume_data:
            raise ValueError("Resume data verification failed after write")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to save resume state: {e}", exc_info=True)
        return False


def load_and_validate_resume_file(resume_file: str) -> Tuple[Optional[Dict], List[str]]:
    """
    Load and validate a resume file.
    
    Returns: (resume_data_or_none, list_of_errors)
    """
    errors = []
    
    try:
        if not os.path.exists(resume_file):
            errors.append("Resume file does not exist")
            return None, errors
            
        with open(resume_file, 'r') as f:
            resume_data = json.load(f)
            
        # Validate the data
        is_valid, validation_errors = ResumeValidator.validate_resume_data(resume_data)
        if not is_valid:
            errors.extend(validation_errors)
            return None, errors
            
        # Check if resume data is recent (within 24 hours)
        if 'timestamp' in resume_data:
            resume_time = datetime.fromisoformat(resume_data['timestamp'].replace('Z', '+00:00'))
            age_hours = (datetime.now() - resume_time).total_seconds() / 3600
            if age_hours > 24:
                logger.warning(f"Resume data is {age_hours:.1f} hours old")
                
        return resume_data, []
        
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in resume file: {e}")
    except Exception as e:
        errors.append(f"Error loading resume file: {e}")
        
    return None, errors