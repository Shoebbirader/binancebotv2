"""Enhanced position monitoring and management"""
import logging
import time
from datetime import datetime
import threading
from typing import Dict, Optional

class PositionMonitor:
    def __init__(self, paper_trader, risk_manager):
        self.paper_trader = paper_trader
        self.risk_manager = risk_manager
        self._monitor_thread = None
        self._monitoring = False
        self._lock = threading.Lock()
        
    def start_monitoring(self, check_interval: int = 30):
        """Start background position monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(check_interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logging.info("Position monitoring started")
    
    def stop_monitoring(self):
        """Stop background position monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("Position monitoring stopped")
    
    def _monitor_loop(self, check_interval: int):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self._check_all_positions()
                time.sleep(check_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _check_all_positions(self):
        """Check all active positions for exit conditions"""
        with self._lock:
            # Get copy of active positions to avoid iteration issues
            active_positions = dict(self.paper_trader.positions)
        
        for symbol in active_positions:
            try:
                self._check_position_exit(symbol)
            except Exception as e:
                logging.error(f"Error checking position {symbol}: {e}")
    
    def _check_position_exit(self, symbol: str):
        """Check if a position should be closed"""
        try:
            # Get current price
            current_price = self.paper_trader.get_current_price(symbol)
            if current_price is None:
                return
            
            # Update position P&L
            self.paper_trader._update_position_pnl(symbol, current_price)
            
            # Check risk manager exit signals
            exit_signal, position = self.risk_manager.update_position(symbol, current_price)
            
            if exit_signal:
                logging.info(f"Exit signal {exit_signal} for {symbol} at {current_price}")
                
                # Close position
                close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                result = self.paper_trader.close_position(symbol, close_side, position['quantity'])
                
                if result.get('status') == 'FILLED':
                    # Update risk manager tracking
                    self.risk_manager.close_position_tracking(symbol, current_price, exit_signal)
                    logging.info(f"Position {symbol} closed successfully: {exit_signal}")
                
        except Exception as e:
            logging.error(f"Error in position exit check for {symbol}: {e}")
    
    def force_close_position(self, symbol: str, reason: str = "Manual close"):
        """Manually close a position"""
        try:
            if symbol not in self.paper_trader.positions:
                logging.warning(f"No position found for {symbol}")
                return False
            
            position = self.paper_trader.positions[symbol]
            current_price = self.paper_trader.get_current_price(symbol)
            
            if current_price is None:
                logging.error(f"Cannot get price for {symbol}")
                return False
            
            close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            result = self.paper_trader.close_position(symbol, close_side, position['quantity'])
            
            if result.get('status') == 'FILLED':
                self.risk_manager.close_position_tracking(symbol, current_price, reason)
                logging.info(f"Position {symbol} manually closed: {reason}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error manually closing position {symbol}: {e}")
            return False
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'monitoring_active': self._monitoring,
            'active_positions': len(self.paper_trader.positions),
            'positions': list(self.paper_trader.positions.keys())
        }

# Global position monitor (will be initialized in main)
position_monitor = None
