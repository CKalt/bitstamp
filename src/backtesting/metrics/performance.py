"""
Performance Metrics Calculator for Backtesting
Provides comprehensive trading performance analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate comprehensive trading performance metrics"""
    
    def __init__(self, initial_capital: float = 100000.0, risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        
    def calculate_all(self, 
                     equity_curve: pd.DataFrame,
                     trades: List[Dict],
                     signals: List[Dict],
                     regime_history: List[Dict]) -> Dict[str, Any]:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: DataFrame with timestamp and portfolio value
            trades: List of trade dictionaries
            signals: List of signal dictionaries
            regime_history: List of market regime classifications
            
        Returns:
            Comprehensive metrics dictionary
        """
        # Convert to DataFrames for easier analysis
        equity_df = pd.DataFrame(equity_curve)
        if 'timestamp' in equity_df.columns:
            equity_df.set_index('timestamp', inplace=True)
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        regime_df = pd.DataFrame(regime_history) if regime_history else pd.DataFrame()
        
        # Calculate returns
        returns = self._calculate_returns(equity_df)
        
        # Basic performance metrics
        metrics = {
            'summary': self._calculate_summary_metrics(equity_df, returns),
            'returns': self._calculate_return_metrics(returns),
            'risk': self._calculate_risk_metrics(returns, equity_df),
            'trading': self._calculate_trading_metrics(trades_df) if not trades_df.empty else {},
            'regime_analysis': self._analyze_by_regime(trades_df, regime_df) if not regime_df.empty else {},
            'drawdown_analysis': self._calculate_drawdown_metrics(equity_df),
            'monthly_returns': self._calculate_monthly_returns(returns),
            'trade_analysis': self._analyze_trades(trades_df) if not trades_df.empty else {},
            'win_loss_analysis': self._calculate_win_loss_metrics(trades_df) if not trades_df.empty else {}
        }
        
        return metrics
    
    def _calculate_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calculate period returns from equity curve"""
        if 'value' in equity_df.columns:
            return equity_df['value'].pct_change().fillna(0)
        else:
            logger.warning("No 'value' column in equity DataFrame")
            return pd.Series()
    
    def _calculate_summary_metrics(self, equity_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """Calculate summary performance metrics"""
        if equity_df.empty or 'value' not in equity_df.columns:
            return {}
        
        initial_value = equity_df['value'].iloc[0]
        final_value = equity_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate time span
        time_span = equity_df.index[-1] - equity_df.index[0]
        years = time_span.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'total_days': time_span.days,
            'total_years': years
        }
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate detailed return metrics"""
        if returns.empty:
            return {}
        
        # Remove any NaN or infinite values
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_returns.empty:
            return {}
        
        return {
            'mean_return': clean_returns.mean(),
            'median_return': clean_returns.median(),
            'std_return': clean_returns.std(),
            'skewness': clean_returns.skew(),
            'kurtosis': clean_returns.kurtosis(),
            'best_day': clean_returns.max(),
            'worst_day': clean_returns.min(),
            'positive_days': (clean_returns > 0).sum(),
            'negative_days': (clean_returns < 0).sum(),
            'win_rate_days': (clean_returns > 0).sum() / len(clean_returns) if len(clean_returns) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series, equity_df: pd.DataFrame) -> Dict:
        """Calculate risk-related metrics"""
        if returns.empty:
            return {}
        
        # Clean returns
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_returns.empty:
            return {}
        
        # Sharpe Ratio (assuming hourly data, annualize appropriately)
        hours_per_year = 365.25 * 24
        periods_per_year = hours_per_year  # Since we have hourly data
        
        excess_returns = clean_returns - (self.risk_free_rate / periods_per_year)
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / clean_returns.std() if clean_returns.std() > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = clean_returns[clean_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(clean_returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = clean_returns[clean_returns <= var_95].mean() if len(clean_returns[clean_returns <= var_95]) > 0 else var_95
        
        # Maximum drawdown
        drawdown_info = self._calculate_drawdown_metrics(equity_df)
        
        # Calmar Ratio
        max_dd = abs(drawdown_info.get('max_drawdown', 0))
        annual_return = self._calculate_summary_metrics(equity_df, returns).get('annualized_return', 0)
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': clean_returns.std(),
            'annualized_volatility': clean_returns.std() * np.sqrt(periods_per_year)
        }
    
    def _calculate_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """Calculate drawdown-related metrics"""
        if equity_df.empty or 'value' not in equity_df.columns:
            return {}
        
        # Calculate running maximum
        running_max = equity_df['value'].expanding().max()
        
        # Calculate drawdown series
        drawdown = (equity_df['value'] - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        
        # Find drawdown start (peak before max drawdown)
        peak_idx = equity_df['value'][:max_drawdown_idx].idxmax()
        
        # Calculate drawdown duration
        if isinstance(max_drawdown_idx, pd.Timestamp) and isinstance(peak_idx, pd.Timestamp):
            drawdown_duration = (max_drawdown_idx - peak_idx).days
        else:
            drawdown_duration = 0
        
        # Recovery analysis
        if max_drawdown < 0:
            # Find recovery point (when equity exceeds previous peak)
            peak_value = equity_df['value'].loc[peak_idx]
            recovery_mask = equity_df['value'][max_drawdown_idx:] >= peak_value
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                recovery_duration = (recovery_idx - max_drawdown_idx).days
            else:
                recovery_duration = None  # Not recovered yet
        else:
            recovery_duration = 0
        
        # Calculate all drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_peak = None
        current_trough = None
        
        for idx, value in equity_df['value'].items():
            if not in_drawdown:
                if current_peak is None or value > current_peak[1]:
                    current_peak = (idx, value)
                elif current_peak and value < current_peak[1] * 0.99:  # 1% threshold
                    in_drawdown = True
                    current_trough = (idx, value)
            else:
                if value < current_trough[1]:
                    current_trough = (idx, value)
                elif value >= current_peak[1]:
                    # Drawdown ended
                    drawdown_pct = (current_trough[1] - current_peak[1]) / current_peak[1]
                    duration = (current_trough[0] - current_peak[0]).days
                    recovery = (idx - current_trough[0]).days
                    drawdown_periods.append({
                        'start': current_peak[0],
                        'end': idx,
                        'drawdown_pct': drawdown_pct,
                        'duration_days': duration,
                        'recovery_days': recovery
                    })
                    in_drawdown = False
                    current_peak = (idx, value)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_date': max_drawdown_idx,
            'drawdown_duration_days': drawdown_duration,
            'recovery_duration_days': recovery_duration,
            'num_drawdowns': len(drawdown_periods),
            'avg_drawdown': np.mean([d['drawdown_pct'] for d in drawdown_periods]) if drawdown_periods else 0,
            'drawdown_periods': drawdown_periods[:5]  # Top 5 drawdowns
        }
    
    def _calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trading-specific metrics"""
        if trades_df.empty:
            return {}
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Calculate trade frequency
        if len(trades_df) > 1:
            time_span = trades_df['timestamp'].max() - trades_df['timestamp'].min()
            days = time_span.total_seconds() / (24 * 60 * 60)
            trades_per_day = len(trades_df) / days if days > 0 else 0
        else:
            trades_per_day = 0
        
        # Calculate fees
        total_fees = trades_df['fee'].sum() if 'fee' in trades_df.columns else 0
        
        # Position changes
        buys = trades_df[trades_df['side'] == 'buy']
        sells = trades_df[trades_df['side'] == 'sell']
        
        return {
            'total_trades': len(trades_df),
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'trades_per_day': trades_per_day,
            'total_fees': total_fees,
            'avg_fee_per_trade': total_fees / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_trade_amount': trades_df['amount'].mean() if 'amount' in trades_df.columns else 0,
            'total_volume': trades_df['amount'].sum() if 'amount' in trades_df.columns else 0
        }
    
    def _calculate_win_loss_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate win/loss metrics from completed round trips"""
        if trades_df.empty or len(trades_df) < 2:
            return {}
        
        # Find round trips (buy followed by sell)
        round_trips = []
        for i in range(len(trades_df) - 1):
            if trades_df.iloc[i]['side'] == 'buy' and trades_df.iloc[i + 1]['side'] == 'sell':
                buy_trade = trades_df.iloc[i]
                sell_trade = trades_df.iloc[i + 1]
                
                # Calculate P&L
                buy_cost = buy_trade['total_cost']
                sell_proceeds = sell_trade['total_cost']  # For sells, this is net proceeds
                pnl = sell_proceeds - buy_cost
                pnl_pct = pnl / buy_cost
                
                # Calculate duration
                duration = (sell_trade['timestamp'] - buy_trade['timestamp']).total_seconds() / 3600
                
                round_trips.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_hours': duration,
                    'buy_price': buy_trade['price'],
                    'sell_price': sell_trade['price']
                })
        
        if not round_trips:
            return {}
        
        # Calculate metrics
        pnls = [rt['pnl'] for rt in round_trips]
        pnl_pcts = [rt['pnl_pct'] for rt in round_trips]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average win/loss ratio
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
        
        # Kelly Criterion
        if win_rate > 0 and win_rate < 1 and avg_win_loss_ratio > 0:
            kelly_pct = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        else:
            kelly_pct = 0
        
        return {
            'round_trips': len(round_trips),
            'win_rate': win_rate,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0,
            'total_pnl': sum(pnls),
            'kelly_criterion': kelly_pct,
            'avg_duration_hours': np.mean([rt['duration_hours'] for rt in round_trips])
        }
    
    def _analyze_by_regime(self, trades_df: pd.DataFrame, regime_df: pd.DataFrame) -> Dict:
        """Analyze performance by market regime"""
        if trades_df.empty or regime_df.empty:
            return {}
        
        # Merge trades with regimes
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        regime_df['timestamp'] = pd.to_datetime(regime_df['timestamp'])
        
        # Find regime for each trade
        trade_regimes = []
        for _, trade in trades_df.iterrows():
            # Find the regime at trade time
            regime_mask = regime_df['timestamp'] <= trade['timestamp']
            if regime_mask.any():
                regime = regime_df.loc[regime_mask, 'regime'].iloc[-1]
            else:
                regime = 'unknown'
            trade_regimes.append(regime)
        
        trades_df['regime'] = trade_regimes
        
        # Analyze by regime
        regime_stats = {}
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_stats[regime] = {
                'trade_count': len(regime_trades),
                'trade_percentage': len(regime_trades) / len(trades_df) * 100,
                'buy_count': len(regime_trades[regime_trades['side'] == 'buy']),
                'sell_count': len(regime_trades[regime_trades['side'] == 'sell'])
            }
        
        # Calculate regime durations
        regime_changes = regime_df[regime_df['regime'].ne(regime_df['regime'].shift())]
        regime_durations = {}
        
        for regime in regime_df['regime'].unique():
            regime_periods = regime_df[regime_df['regime'] == regime]
            if not regime_periods.empty:
                total_hours = len(regime_periods)  # Assuming hourly data
                regime_durations[regime] = {
                    'total_hours': total_hours,
                    'percentage': total_hours / len(regime_df) * 100
                }
        
        return {
            'trade_distribution': regime_stats,
            'regime_durations': regime_durations
        }
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> Dict:
        """Calculate monthly return statistics"""
        if returns.empty:
            return {}
        
        # Ensure we have a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return {}
        
        # Calculate monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        if monthly_returns.empty:
            return {}
        
        # Calculate statistics
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        
        return {
            'total_months': len(monthly_returns),
            'positive_months': positive_months,
            'negative_months': negative_months,
            'win_rate_monthly': positive_months / len(monthly_returns) if len(monthly_returns) > 0 else 0,
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_volatility': monthly_returns.std()
        }
    
    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict:
        """Detailed trade analysis"""
        if trades_df.empty:
            return {}
        
        # Time-based analysis
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day_of_week'] = trades_df['timestamp'].dt.dayofweek
        
        # Hour distribution
        hour_dist = trades_df['hour'].value_counts().sort_index().to_dict()
        
        # Day of week distribution (0=Monday, 6=Sunday)
        dow_dist = trades_df['day_of_week'].value_counts().sort_index().to_dict()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_dist_named = {day_names[k]: v for k, v in dow_dist.items()}
        
        # Trade gaps
        if len(trades_df) > 1:
            trade_gaps = trades_df['timestamp'].diff().dropna()
            avg_gap_hours = trade_gaps.mean().total_seconds() / 3600
            min_gap_hours = trade_gaps.min().total_seconds() / 3600
            max_gap_hours = trade_gaps.max().total_seconds() / 3600
        else:
            avg_gap_hours = min_gap_hours = max_gap_hours = 0
        
        return {
            'hour_distribution': hour_dist,
            'day_of_week_distribution': dow_dist_named,
            'avg_time_between_trades_hours': avg_gap_hours,
            'min_time_between_trades_hours': min_gap_hours,
            'max_time_between_trades_hours': max_gap_hours
        }
    
    def generate_summary_report(self, metrics: Dict) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append("=== BACKTEST PERFORMANCE SUMMARY ===\n")
        
        # Summary section
        if 'summary' in metrics:
            s = metrics['summary']
            report.append(f"Initial Capital: ${s.get('initial_value', 0):,.2f}")
            report.append(f"Final Value: ${s.get('final_value', 0):,.2f}")
            report.append(f"Total Return: {s.get('total_return_pct', 0):.2f}%")
            report.append(f"Annualized Return: {s.get('annualized_return_pct', 0):.2f}%")
            report.append(f"Trading Period: {s.get('total_days', 0)} days\n")
        
        # Risk metrics
        if 'risk' in metrics:
            r = metrics['risk']
            report.append("Risk Metrics:")
            report.append(f"  Sharpe Ratio: {r.get('sharpe_ratio', 0):.3f}")
            report.append(f"  Sortino Ratio: {r.get('sortino_ratio', 0):.3f}")
            report.append(f"  Calmar Ratio: {r.get('calmar_ratio', 0):.3f}")
            report.append(f"  Annual Volatility: {r.get('annualized_volatility', 0)*100:.2f}%")
            report.append(f"  Value at Risk (95%): {r.get('var_95', 0)*100:.2f}%\n")
        
        # Drawdown
        if 'drawdown_analysis' in metrics:
            d = metrics['drawdown_analysis']
            report.append("Drawdown Analysis:")
            report.append(f"  Max Drawdown: {d.get('max_drawdown_pct', 0):.2f}%")
            report.append(f"  Drawdown Duration: {d.get('drawdown_duration_days', 0)} days")
            report.append(f"  Recovery Duration: {d.get('recovery_duration_days', 'N/A')} days\n")
        
        # Trading metrics
        if 'trading' in metrics:
            t = metrics['trading']
            report.append("Trading Activity:")
            report.append(f"  Total Trades: {t.get('total_trades', 0)}")
            report.append(f"  Trades per Day: {t.get('trades_per_day', 0):.2f}")
            report.append(f"  Total Fees Paid: ${t.get('total_fees', 0):.2f}\n")
        
        # Win/Loss metrics
        if 'win_loss_analysis' in metrics:
            w = metrics['win_loss_analysis']
            report.append("Win/Loss Analysis:")
            report.append(f"  Win Rate: {w.get('win_rate', 0)*100:.1f}%")
            report.append(f"  Profit Factor: {w.get('profit_factor', 0):.2f}")
            report.append(f"  Avg Win/Loss Ratio: {w.get('avg_win_loss_ratio', 0):.2f}")
            report.append(f"  Average P&L per Trade: ${w.get('avg_pnl', 0):.2f}")
            report.append(f"  Kelly Criterion: {w.get('kelly_criterion', 0)*100:.1f}%\n")
        
        return "\n".join(report)