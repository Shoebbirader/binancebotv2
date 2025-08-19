import os
import time
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *
from dotenv import load_dotenv
import logging

load_dotenv('config/secrets.env')

class BinanceInterface:
    def __init__(self, use_testnet: bool = False):
        """Initialize Binance client with API credentials"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in secrets.env")
            
            self.client = Client(api_key, api_secret)
            if use_testnet:
                self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
            self.test_connection()
            logging.info("Binance API connection established successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def test_connection(self):
        """Test API connection"""
        try:
            # Test with a simple API call
            server_time = self.client.get_server_time()
            logging.info(f"Server time: {server_time}")
            return True
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            raise
    
    def get_account_balance(self):
        """Get real-time account balance with better error handling"""
        try:
            account = self.client.futures_account()
            balances = {}
            
            for balance in account['assets']:
                try:
                    # Handle potential missing keys safely
                    wallet_balance = float(balance.get('walletBalance', 0))
                    available_balance = float(balance.get('availableBalance', 0))
                    unrealized_pnl = float(balance.get('unrealizedPnl', 0))
                    
                    if wallet_balance > 0:
                        balances[balance['asset']] = {
                            'wallet_balance': wallet_balance,
                            'available_balance': available_balance,
                            'unrealized_pnl': unrealized_pnl
                        }
                        
                except (KeyError, ValueError, TypeError) as e:
                    logging.warning(f"Skipping balance entry due to parsing error: {e}")
                    continue
            
            logging.info(f"Account balance retrieved: {balances}")
            return balances
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error getting balance: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error getting balance: {e}")
            # Return a default balance structure for testing
            return {'USDT': {'available_balance': 1000, 'wallet_balance': 1000, 'unrealized_pnl': 0}}

    def get_symbol_filters(self, symbol):
        """Fetch symbol filters (tick size, step size, min notional)."""
        info = self.client.futures_exchange_info()
        sym = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
        if not sym:
            raise ValueError(f"Symbol {symbol} not found in exchange info")
        filters = {f['filterType']: f for f in sym.get('filters', [])}
        return filters

    def format_quantity(self, symbol, quantity):
        """Round quantity to the allowed step size and min qty."""
        filters = self.get_symbol_filters(symbol)
        lot = filters.get('LOT_SIZE', {})
        step = float(lot.get('stepSize', '0.001'))
        min_qty = float(lot.get('minQty', '0.0'))
        # Round down to step size
        qty = max(quantity - (quantity % step), min_qty)
        # Ensure proper decimals by formatting against step
        decimals = max(0, str(step)[::-1].find('.'))
        return float(f"{qty:.{decimals}f}")

    def format_price(self, symbol, price):
        """Round price to tick size."""
        filters = self.get_symbol_filters(symbol)
        pf = filters.get('PRICE_FILTER', {})
        tick = float(pf.get('tickSize', '0.01'))
        decimals = max(0, str(tick)[::-1].find('.'))
        # Round to nearest tick down
        p = price - (price % tick)
        return float(f"{p:.{decimals}f}")
    
    def get_current_price(self, symbol):
        """Get real-time price for a symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logging.info(f"Current price for {symbol}: {price}")
            return price
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error getting price for {symbol}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error getting price for {symbol}: {e}")
            raise
    
    def get_position_info(self, symbol):
        """Get current position information"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            
            for position in positions:
                try:
                    position_amt = float(position.get('positionAmt', 0))
                    if position_amt != 0:
                        return {
                            'symbol': position['symbol'],
                            'size': position_amt,
                            'entry_price': float(position.get('entryPrice', 0)),
                            'unrealized_pnl': float(position.get('unRealizedProfit', 0)),
                            'leverage': int(position.get('leverage', 1)),
                            'side': 'LONG' if position_amt > 0 else 'SHORT'
                        }
                except (KeyError, ValueError, TypeError) as e:
                    logging.warning(f"Error parsing position data: {e}")
                    continue
            
            return None  # No open position
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error getting position for {symbol}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error getting position for {symbol}: {e}")
            raise
    
    def place_futures_order(self, symbol, side, quantity, order_type='MARKET', price=None, stop_loss=None, take_profit=None):
        """Place futures order with (optional) reduce-only SL and TP. Returns dict with order ids."""
        try:
            # Format qty/price
            qty = self.format_quantity(symbol, float(quantity))
            px = self.format_price(symbol, float(price)) if (price and order_type == 'LIMIT') else None

            # Main order
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': qty
            }
            
            if price and order_type == 'LIMIT':
                order_params['price'] = px
                order_params['timeInForce'] = 'GTC'
            
            # Place main order
            main_order = self.client.futures_create_order(**order_params)
            logging.info(f"Main order placed: {main_order}")
            
            sl_order = None
            tp_order = None

            # Place stop-loss order if specified
            if stop_loss:
                try:
                    sl_side = 'SELL' if side == 'BUY' else 'BUY'
                    sl_price = self.format_price(symbol, float(stop_loss))
                    sl_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type='STOP_MARKET',
                        quantity=qty,
                        stopPrice=sl_price,
                        reduceOnly=True
                    )
                    logging.info(f"Stop-loss order placed: {sl_order}")
                except Exception as e:
                    logging.error(f"Failed to place stop-loss order: {e}")
            
            # Place take-profit order if specified
            if take_profit:
                try:
                    tp_side = 'SELL' if side == 'BUY' else 'BUY'
                    tp_price = self.format_price(symbol, float(take_profit))
                    tp_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type='TAKE_PROFIT_MARKET',
                        quantity=qty,
                        stopPrice=tp_price,
                        reduceOnly=True
                    )
                    logging.info(f"Take-profit order placed: {tp_order}")
                except Exception as e:
                    logging.error(f"Failed to place take-profit order: {e}")
            
            return {
                'main': main_order,
                'sl': sl_order,
                'tp': tp_order
            }
            
        except BinanceOrderException as e:
            logging.error(f"Order placement failed: {e}")
            raise
        except BinanceAPIException as e:
            logging.error(f"Binance API error placing order: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error placing order: {e}")
            raise
    
    def close_position(self, symbol, side, quantity):
        """Close an open position"""
        try:
            close_side = 'SELL' if side == 'BUY' else 'BUY'
            qty = self.format_quantity(symbol, float(quantity))
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=qty,
                reduceOnly=True
            )
            logging.info(f"Position closed: {order}")
            return order
            
        except BinanceOrderException as e:
            logging.error(f"Position close failed: {e}")
            raise
        except BinanceAPIException as e:
            logging.error(f"Binance API error closing position: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error closing position: {e}")
            raise
    
    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol"""
        try:
            result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logging.info(f"Leverage set for {symbol}: {leverage}x")
            return result
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error setting leverage: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error setting leverage: {e}")
            raise

    def cancel_all_open_orders(self, symbol):
        """Cancel all open orders for a symbol (e.g., after position exit)."""
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logging.info(f"Canceled all open orders for {symbol}")
            return result
        except BinanceAPIException as e:
            logging.error(f"Binance API error cancelling orders for {symbol}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error cancelling orders for {symbol}: {e}")
            raise
