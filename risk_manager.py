import numpy as np
import logging
from datetime import datetime, timedelta
import threading

class RiskManager:
    def __init__(self, config):
        self.max_loss_pct = config['max_loss_per_trade_pct']
        self.position_size_pct = config['position_size_pct']
        self.min_leverage = config['min_leverage']
        self.max_leverage = config['max_leverage']
        self.stop_loss_percent = config.get('stop_loss_percent', 5)
        self.take_profit_percent = config.get('take_profit_percent', 15)
        self.max_positions = config.get('max_positions', 3)
        self.cooldown_seconds = config.get('cooldown_seconds', 60)
        
        # Position tracking
        self.active_positions = {}
        self.position_history = []
        # Thread-safety
        self._lock = threading.Lock()
        self._last_entry_times = {}
        
    def calculate_position_size(self, balance, price, leverage):
        """Enhanced position sizing with risk management"""
        try:
            # Base position size
            raw_qty = (self.position_size_pct / 100.0) * balance * leverage / price
            
            # Risk-adjusted sizing based on volatility
            risk_adjusted_qty = raw_qty * 0.8  # Conservative approach
            
            quantity = round(risk_adjusted_qty, 3)
            final_qty = max(quantity, 0.001)
            
            logging.info(f"Position size calculated: {final_qty:.6f} (Risk adjusted from {raw_qty:.6f})")
            return final_qty
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.001  # Minimum safe quantity
    
    def should_enter(self, pred_accuracy, threshold):
        """Enhanced entry criteria with confidence bands"""
        try:
            # Add confidence bands for better decision making
            if pred_accuracy > threshold + 0.1:  # High confidence
                return True, "HIGH"
            elif pred_accuracy > threshold:  # Medium confidence
                return True, "MEDIUM"
            else:  # Low confidence
                return False, "LOW"
                
        except Exception as e:
            logging.error(f"Error in entry criteria: {e}")
            return False, "ERROR"
    
    def calculate_stop_loss_price(self, entry_price, side, stop_loss_pct):
        """Calculate stop-loss price"""
        try:
            if side == 'BUY':
                stop_loss = entry_price * (1 - stop_loss_pct / 100.0)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_loss_pct / 100.0)
            
            return round(stop_loss, 6)
            
        except Exception as e:
            logging.error(f"Error calculating stop-loss: {e}")
            return None
    
    def calculate_take_profit_price(self, entry_price, side, take_profit_pct):
        """Calculate take-profit price"""
        try:
            if side == 'BUY':
                take_profit = entry_price * (1 + take_profit_pct / 100.0)
            else:  # SELL
                take_profit = entry_price * (1 - take_profit_pct / 100.0)
            
            return round(take_profit, 6)
            
        except Exception as e:
            logging.error(f"Error calculating take-profit: {e}")
            return None
    
    def should_stop_loss(self, entry_price, current_price, side):
        """Check if stop-loss should be triggered"""
        try:
            if side == 'BUY':
                loss_pct = (entry_price - current_price) / entry_price * 100
            else:  # SELL
                loss_pct = (current_price - entry_price) / entry_price * 100
                
            return loss_pct >= self.stop_loss_percent
            
        except Exception as e:
            logging.error(f"Error checking stop-loss: {e}")
            return False
    
    def should_take_profit(self, entry_price, current_price, side):
        """Check if take-profit should be triggered"""
        try:
            if side == 'BUY':
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:  # SELL
                profit_pct = (entry_price - current_price) / entry_price * 100
                
            return profit_pct >= self.take_profit_percent
            
        except Exception as e:
            logging.error(f"Error checking take-profit: {e}")
            return False
    
    def add_position(self, symbol, side, entry_price, quantity, leverage, stop_loss, take_profit, bracket_orders=None):
        """Add new position to tracking"""
        try:
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'status': 'ACTIVE',
                'bracket_orders': bracket_orders or {}
            }
            
            with self._lock:
                self.active_positions[symbol] = position
                self._last_entry_times[symbol] = datetime.now()
            logging.info(f"Position added to tracking: {symbol} {side} {quantity}")
            
        except Exception as e:
            logging.error(f"Error adding position: {e}")
    
    def update_position(self, symbol, current_price):
        """Update position and check for exit signals"""
        try:
            with self._lock:
                if symbol not in self.active_positions:
                    return None, None
                position = dict(self.active_positions[symbol])
            entry_price = position['entry_price']
            side = position['side']
            
            # Check stop-loss
            if self.should_stop_loss(entry_price, current_price, side):
                logging.info(f"Stop-loss triggered for {symbol} at {current_price}")
                return 'STOP_LOSS', position
            
            # Check take-profit
            if self.should_take_profit(entry_price, current_price, side):
                logging.info(f"Take-profit triggered for {symbol} at {current_price}")
                return 'TAKE_PROFIT', position
            
            return None, None
            
        except Exception as e:
            logging.error(f"Error updating position: {e}")
            return None, None
    
    def close_position_tracking(self, symbol, exit_price, exit_reason):
        """Close position in tracking"""
        try:
            with self._lock:
                if symbol in self.active_positions:
                    position = self.active_positions[symbol]
                    position['exit_price'] = exit_price
                    position['exit_time'] = datetime.now()
                    position['exit_reason'] = exit_reason
                    position['status'] = 'CLOSED'
                    
                    # Calculate P&L
                    if position['side'] == 'BUY':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                    
                    position['pnl_pct'] = pnl_pct
                    
                    # Move to history
                    self.position_history.append(position)
                    del self.active_positions[symbol]
                
                logging.info(f"Position closed: {symbol} {exit_reason} P&L: {pnl_pct:.2f}%")
                
        except Exception as e:
            logging.error(f"Error closing position tracking: {e}")
    
    def get_position_summary(self):
        """Get summary of all positions"""
        try:
            with self._lock:
                active_count = len(self.active_positions)
                total_pnl = sum([pos.get('pnl_pct', 0) for pos in self.position_history])
            
            return {
                'active_positions': active_count,
                'total_positions': len(self.position_history),
                'total_pnl_pct': total_pnl,
                'active_symbols': list(self.active_positions.keys())
            }
            
        except Exception as e:
            logging.error(f"Error getting position summary: {e}")
            return {}

    def can_open_new_position(self, symbol):
        """Check if we can open a new position considering max positions and cooldown."""
        try:
            with self._lock:
                # Enforce max total positions
                if len(self.active_positions) >= self.max_positions:
                    return False, "MAX_POSITIONS"
                # Avoid duplicate symbol
                if symbol in self.active_positions:
                    return False, "ALREADY_OPEN"
                # Cooldown per symbol
                last_time = self._last_entry_times.get(symbol)
                if last_time is not None:
                    elapsed = (datetime.now() - last_time).total_seconds()
                    if elapsed < self.cooldown_seconds:
                        return False, "COOLDOWN"
            return True, "OK"
        except Exception as e:
            logging.error(f"Error in can_open_new_position: {e}")
            return False, "ERROR"
