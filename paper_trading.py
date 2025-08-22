import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import threading

class PaperTradingInterface:
    """
    Paper trading simulation for Binance Futures
    Simulates real trading without using real money
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.order_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.current_prices: Dict[str, float] = {}
        self.pnl_history: List[float] = []
        self._lock = threading.Lock()
        logging.info(f"Paper trading initialized with ${initial_balance:.2f} USDT")
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Simulate setting leverage in a thread-safe way"""
        with self._lock:
            if not hasattr(self, '_leverage_map'):
                self._leverage_map = {}
            self._leverage_map[symbol] = leverage
            logging.info(f"Paper trading: Leverage set to {leverage}x for {symbol}")
        return True
    
    def get_account_balance(self) -> Dict[str, Dict]:
        """Get simulated account balance"""
        with self._lock:
            return {
                'USDT': {
                    'available_balance': self.current_balance,
                    'wallet_balance': self.current_balance,
                    'total_balance': self.current_balance,
                    'unrealized_pnl': sum([pos.get('unrealized_pnl', 0) for pos in self.positions.values()])
                }
            }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance API for paper trading"""
        try:
            # For paper trading, we should still use real market prices
            from binance.client import Client
            import os
            if not hasattr(self, '_binance_client'):
                self._binance_client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
            
            # Get real price from Binance
            ticker = self._binance_client.futures_mark_price(symbol=symbol)
            price = float(ticker['markPrice'])
            
            # Validate price
            if np.isnan(price) or np.isinf(price) or price <= 0:
                logging.error(f"Invalid real price for {symbol}: {price}")
                return None
                
            self.current_prices[symbol] = price
            return price
            
        except Exception as e:
            logging.error(f"Failed to get real price for {symbol}: {e}")
            # Only use fallback if real price fails
            price = self.current_prices.get(symbol)
            if price is None:
                logging.warning(f"No cached price for {symbol}, paper trading may be inaccurate")
                return None
            return price
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for simulation"""
        self.current_prices.update(prices)
        
        # Update unrealized P&L for active positions
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                self._update_position_pnl(symbol, current_price)
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get simulated position info"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return {
                'symbol': symbol,
                'size': pos['quantity'] if pos['side'] == 'LONG' else -pos['quantity'],
                'entry_price': pos['entry_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'leverage': pos['leverage'],
                'side': 'LONG' if pos['side'] == 'LONG' else 'SHORT'
            }
        return None
    
    def place_futures_order(self, symbol: str, side: str, quantity: float, 
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Dict:
        """Simulate placing a futures order"""
        with self._lock:
            if symbol not in self.current_prices:
                raise ValueError(f"No price available for {symbol}")
            current_price = self.current_prices[symbol]
            order_value = quantity * current_price
            # Get leverage from the leverage map if available, otherwise use 1
            leverage = getattr(self, '_leverage_map', {}).get(symbol, 1)
            margin_required = order_value / leverage
            
            if margin_required > self.current_balance:
                raise ValueError(f"Insufficient paper trading balance: {self.current_balance:.2f} USDT for margin required {margin_required:.2f}")
            
            # For paper trading, we only need to reserve the margin, not the full order value
            self.current_balance -= margin_required
            position_margin = margin_required
            
            logging.info(f"Paper Trade: Reserved ${margin_required:.2f} margin (${leverage}x leverage), Remaining balance: ${self.current_balance:.2f}")
            
            # Create order
            order = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'orderId': f"paper_{int(time.time() * 1000)}",
                'status': 'FILLED',
                'time': datetime.now().isoformat(),
                'entry_price': current_price
            }
            
            # Create position
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'LONG' if side == 'BUY' else 'SHORT',
                'quantity': quantity,
                'entry_price': current_price,
                'leverage': leverage,
                'margin_used': position_margin,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'unrealized_pnl': 0.0
            }
            
            self.order_history.append(order)
            logging.info(f"Paper trade: {side} {quantity} {symbol} @ ${current_price:.4f}")
            # Log to CSV
            self._log_trade_csv(order)
            return {
                'main': order,
                'sl': {'orderId': f"sl_{order['orderId']}"} if stop_loss else None,
                'tp': {'orderId': f"tp_{order['orderId']}"} if take_profit else None
            }
    
    def close_position(self, symbol: str, side: str, quantity: float) -> Dict:
        """Simulate closing a position with thread safety"""
        with self._lock:
            if symbol not in self.positions:
                raise ValueError(f"No position found for {symbol}")
            position = self.positions[symbol]
            current_price = self.current_prices.get(symbol)
            if current_price is None:
                raise ValueError(f"No price available for {symbol}")
            # Calculate P&L with leverage
            if position['side'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['quantity'] * position.get('leverage', 1)
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity'] * position.get('leverage', 1)
            # Return margin to balance and add/subtract P&L
            margin_used = position.get('margin_used', (position['entry_price'] * position['quantity']) / position.get('leverage', 1))
            self.current_balance += margin_used + pnl
            # Record trade
            trade = {
                'symbol': symbol,
                'side': 'SELL' if position['side'] == 'LONG' else 'BUY',
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'exit_time': datetime.now().isoformat()
            }
            self.trade_history.append(trade)
            del self.positions[symbol]
            self.pnl_history.append(pnl)
            logging.info(f"Paper trade closed: {symbol} P&L: ${pnl:.4f}, Balance: ${self.current_balance:.2f}")
            self._log_trade_csv(trade)
            return {'orderId': f"close_{int(time.time() * 1000)}", 'status': 'FILLED'}
    
    def _update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for active positions with thread safety"""
        # First check if position exists without holding lock too long
        if symbol not in self.positions:
            return
            
        # Get position info atomically
        with self._lock:
            if symbol not in self.positions:
                return
            position = dict(self.positions[symbol])  # Create copy to work with
            
        # Calculate P&L outside of lock
        if position['side'] == 'LONG':
            pnl = (current_price - position['entry_price']) * position['quantity'] * position.get('leverage', 1)
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity'] * position.get('leverage', 1)
            
        # Update P&L atomically
        with self._lock:
            if symbol in self.positions:
                self.positions[symbol]['unrealized_pnl'] = pnl
                
        # Check exit conditions and close if needed (this calls close_position which has its own lock)
        if position.get('stop_loss'):
            if position['side'] == 'LONG' and current_price <= position['stop_loss']:
                self.close_position(symbol, 'SELL', position['quantity'])
                logging.info(f"Paper trading: Stop loss triggered for {symbol}")
            elif position['side'] == 'SHORT' and current_price >= position['stop_loss']:
                self.close_position(symbol, 'BUY', position['quantity'])
                logging.info(f"Paper trading: Stop loss triggered for {symbol}")
                
        if position.get('take_profit'):
            if position['side'] == 'LONG' and current_price >= position['take_profit']:
                self.close_position(symbol, 'SELL', position['quantity'])
                logging.info(f"Paper trading: Take profit triggered for {symbol}")
            elif position['side'] == 'SHORT' and current_price <= position['take_profit']:
                self.close_position(symbol, 'BUY', position['quantity'])
                logging.info(f"Paper trading: Take profit triggered for {symbol}")

    def log_latency(self, start_time, end_time):
        """Simulate latency logging for paper trading"""
        latency_ms = (end_time - start_time) * 1000
        logging.info(f"Paper trading simulated latency: {latency_ms:.2f} ms")
        return latency_ms

    def analyze_slippage(self, symbol, expected_price, executed_order):
        """Simulate slippage analysis for paper trading"""
        logging.info(f"Paper trading: No slippage for {symbol} (simulated execution)")
        return 0

    def _log_trade_csv(self, trade_dict):
        """Disabled - CSV logging handled by main logger to avoid conflicts"""
        pass

    # Advanced order execution stubs
    def execute_vwap(self, symbol, side, quantity, data, order_type='MARKET'):
        return self.place_futures_order(symbol, side, quantity)
    def execute_twap(self, symbol, side, quantity, intervals=3, interval_sec=5, order_type='MARKET'):
        return self.place_futures_order(symbol, side, quantity)
    def smart_order_routing(self, symbol, side, quantity, order_type='MARKET', price=None):
        return self.place_futures_order(symbol, side, quantity)
    def place_trailing_stop_order(self, symbol, side, quantity, activation_price, callback_rate):
        return self.place_futures_order(symbol, side, quantity)
    def place_oco_order(self, symbol, side, quantity, take_profit, stop_loss, stop_limit_price):
        return self.place_futures_order(symbol, side, quantity)
    def place_scaled_orders(self, symbol, side, total_quantity, price_levels):
        orders = []
        for price, qty in price_levels:
            order = self.place_futures_order(symbol, side, qty)
            if order and 'main' in order and order['main']:
                order['main']['price'] = price
            orders.append(order['main'] if order and 'main' in order else None)
        return {'main': orders[0] if orders else None, 'scaled': orders}

# Global paper trading instance
paper_trader = None

def init_paper_trading(initial_balance: float = 10000.0):
    """Initialize global paper trading instance"""
    global paper_trader
    paper_trader = PaperTradingInterface(initial_balance)
    return paper_trader

def get_paper_trader():
    """Get the global paper trading instance"""
    global paper_trader
    if paper_trader is None:
        paper_trader = PaperTradingInterface()
    return paper_trader