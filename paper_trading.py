import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

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
        
        logging.info(f"Paper trading initialized with ${initial_balance:.2f} USDT")
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Simulate setting leverage"""
        logging.info(f"Paper trading: Leverage set to {leverage}x for {symbol}")
        return True
    
    def get_account_balance(self) -> Dict[str, Dict]:
        """Get simulated account balance"""
        return {
            'USDT': {
                'available_balance': self.current_balance,
                'total_balance': self.current_balance
            }
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (simulated); fallback to random price if missing"""
        price = self.current_prices.get(symbol)
        if price is None:
            # Fallback: simulate price between 10 and 100
            price = float(np.random.uniform(10, 100))
            self.current_prices[symbol] = price
            logging.warning(f"No price available for {symbol}, using simulated price: {price:.2f}")
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
        try:
            if symbol not in self.current_prices:
                raise ValueError(f"No price available for {symbol}")
            
            current_price = self.current_prices[symbol]
            order_value = quantity * current_price
            
            # Check if we have enough balance
            if order_value > self.current_balance * 10:  # Allow up to 10x leverage
                raise ValueError("Insufficient balance")
            
            # Create order
            order = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'orderId': f"paper_{int(time.time() * 1000)}",
                'status': 'FILLED',
                'time': datetime.now().isoformat()
            }
            
            # Create position
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'LONG' if side == 'BUY' else 'SHORT',
                'quantity': quantity,
                'entry_price': current_price,
                'leverage': 1,  # Default leverage
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            self.order_history.append(order)
            logging.info(f"Paper trade: {side} {quantity} {symbol} @ ${current_price:.4f}")
            
            return {
                'main': order,
                'sl': {'orderId': f"sl_{order['orderId']}"} if stop_loss else None,
                'tp': {'orderId': f"tp_{order['orderId']}"} if take_profit else None
            }
            
        except Exception as e:
            logging.error(f"Paper trading order failed: {e}")
            raise
    
    def close_position(self, symbol: str, side: str, quantity: float) -> Dict:
        """Simulate closing a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        current_price = self.current_prices.get(symbol)
        
        if current_price is None:
            raise ValueError(f"No price available for {symbol}")
        
        # Calculate P&L
        if position['side'] == 'LONG':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            pnl = (position['entry_price'] - current_price) * position['quantity']
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade
        trade = {
            'symbol': symbol,
            'side': 'SELL' if position['side'] == 'LONG' else 'BUY',
            'quantity': quantity,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pnl': pnl,
            'exit_time': datetime.now().isoformat()
        }
        
        self.trade_history.append(trade)
        del self.positions[symbol]
        self.pnl_history.append(pnl)
        
        logging.info(f"Paper trade closed: {symbol} P&L: ${pnl:.4f}, Balance: ${self.current_balance:.2f}")
        
        return {'orderId': f"close_{int(time.time() * 1000)}", 'status': 'FILLED'}
    
    def cancel_all_open_orders(self, symbol: str) -> bool:
        """Simulate canceling all open orders"""
        logging.info(f"Paper trading: Canceling all orders for {symbol}")
        return True
    
    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        return {
            'balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.current_balance - self.initial_balance,
            'total_pnl_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'active_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'unrealized_pnl': total_unrealized_pnl,
            'positions': list(self.positions.values()),
            'trade_history': self.trade_history[-10:]  # Last 10 trades
        }
    
    def _update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for active positions"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if position['side'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['unrealized_pnl'] = pnl
            
            # Check stop loss and take profit
            if position.get('stop_loss') and position['side'] == 'LONG' and current_price <= position['stop_loss']:
                self.close_position(symbol, 'SELL', position['quantity'])
                logging.info(f"Paper trading: Stop loss triggered for {symbol}")
            elif position.get('stop_loss') and position['side'] == 'SHORT' and current_price >= position['stop_loss']:
                self.close_position(symbol, 'BUY', position['quantity'])
                logging.info(f"Paper trading: Stop loss triggered for {symbol}")
            elif position.get('take_profit') and position['side'] == 'LONG' and current_price >= position['take_profit']:
                self.close_position(symbol, 'SELL', position['quantity'])
                logging.info(f"Paper trading: Take profit triggered for {symbol}")
            elif position.get('take_profit') and position['side'] == 'SHORT' and current_price <= position['take_profit']:
                self.close_position(symbol, 'BUY', position['quantity'])
                logging.info(f"Paper trading: Take profit triggered for {symbol}")

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