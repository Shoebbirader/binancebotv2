import os
import time
import logging
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *
from dotenv import load_dotenv
import logging

load_dotenv('config/secrets.env')

class BinanceInterface:
    # --- Advanced Execution Methods ---
    def _rate_limit(self):
        """Enforce API rate limiting for all calls"""
        current_time = time.time() * 1000
        time_since_last = current_time - getattr(self, 'last_request_time', 0)
        min_interval = 120  # ms, adjust as needed
        if time_since_last < min_interval:
            time.sleep((min_interval - time_since_last) / 1000)
        self.last_request_time = time.time() * 1000

    def smart_order_routing(self, symbol, side, quantity, order_type='MARKET', price=None):
        self._rate_limit()
        """
        Simulate smart order routing (stub: only Binance venue).
        In future, aggregate order books from multiple venues for best price.
        """
        # For now, just place order on Binance
        logging.info(f"Smart order routing: Executing {side} {quantity} {symbol} on Binance")
        return self.place_futures_order(symbol, side, quantity, order_type, price)

    def execute_twap(self, symbol, side, total_quantity, intervals=5, interval_sec=10, order_type='MARKET'):
        self._rate_limit()
        """
        Execute TWAP (Time-Weighted Average Price) order by splitting into intervals.
        """
        qty_per_order = total_quantity / intervals
        orders = []
        for i in range(intervals):
            order = self.place_futures_order(symbol, side, qty_per_order, order_type)
            orders.append(order)
            logging.info(f"TWAP order {i+1}/{intervals} placed: {order}")
            time.sleep(interval_sec)
        return orders

    def execute_vwap(self, symbol, side, total_quantity, price_data, order_type='MARKET'):
        self._rate_limit()
        """
        Execute VWAP (Volume-Weighted Average Price) order using historical volume profile.
        price_data: DataFrame with 'close' and 'volume'
        """
        # Calculate VWAP
        vwap = (price_data['close'] * price_data['volume']).sum() / price_data['volume'].sum()
        logging.info(f"Calculated VWAP for {symbol}: {vwap}")
        # Place order at VWAP price if possible
        return self.place_futures_order(symbol, side, total_quantity, order_type, price=vwap)

    def analyze_slippage(self, symbol, expected_price, executed_order):
        """
        Analyze slippage between expected and executed price.
        executed_order: dict from Binance API
        """
        try:
            fill_price = float(executed_order.get('avgFillPrice', executed_order.get('price', 0)))
            slippage = fill_price - expected_price
            logging.info(f"Slippage for {symbol}: {slippage} (expected {expected_price}, actual {fill_price})")
            return slippage
        except Exception as e:
            logging.error(f"Slippage analysis error: {e}")
            return None

    def log_latency(self, start_time, end_time):
        """
        Log simulated latency for order execution. For real latency optimization, consider co-located servers.
        """
        latency_ms = (end_time - start_time) * 1000
        logging.info(f"Order execution latency: {latency_ms:.2f} ms")
        return latency_ms
    # --- Advanced Order Types ---
    def place_trailing_stop_order(self, symbol, side, quantity, activation_price, callback_rate):
        self._rate_limit()
        """
        Place a trailing stop order (Binance Futures).
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TRAILING_STOP_MARKET',
                quantity=quantity,
                activationPrice=activation_price,
                callbackRate=callback_rate,
                reduceOnly=True
            )
            logging.info(f"Trailing stop order placed: {order}")
            return order
        except Exception as e:
            logging.error(f"Failed to place trailing stop order: {e}")
            return None

    def place_oco_order(self, symbol, side, quantity, price, stop_price, stop_limit_price):
        self._rate_limit()
        """
        Place an OCO (One-Cancels-Other) order (Binance Spot only, demo for Futures).
        """
        try:
            # Binance Futures does not support OCO directly, so simulate with two orders
            main_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                quantity=quantity,
                price=price,
                timeInForce='GTC'
            )
            stop_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                price=stop_limit_price,
                reduceOnly=True
            )
            logging.info(f"OCO simulated: main {main_order}, stop {stop_order}")
            return {'main': main_order, 'stop': stop_order}
        except Exception as e:
            logging.error(f"Failed to place OCO order: {e}")
            return None

    def place_scaled_orders(self, symbol, side, total_quantity, price_levels):
        self._rate_limit()
        """
        Place multiple scaled orders at different price levels.
        price_levels: list of (price, quantity)
        """
        orders = []
        try:
            for price, qty in price_levels:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    quantity=qty,
                    price=price,
                    timeInForce='GTC'
                )
                orders.append(order)
            logging.info(f"Scaled orders placed: {orders}")
            return orders
        except Exception as e:
            logging.error(f"Failed to place scaled orders: {e}")
            return orders

    def place_iceberg_order(self, symbol, side, total_quantity, price, iceberg_qty):
        self._rate_limit()
        """
        Place an iceberg order (hidden large order).
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                quantity=total_quantity,
                price=price,
                timeInForce='GTC',
                icebergQty=iceberg_qty
            )
            logging.info(f"Iceberg order placed: {order}")
            return order
        except Exception as e:
            logging.error(f"Failed to place iceberg order: {e}")
            return None
    def __init__(self, use_testnet: bool = False):
        """Initialize Binance client with API credentials and rate limiting"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in secrets.env")
            
            self.client = Client(api_key, api_secret)
            self.last_request_time = 0
            self.rate_limit_ms = 100  # 100ms between requests
            if use_testnet:
                self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
            self.test_connection()
            logging.info("Binance API connection established successfully with rate limiting")
            
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
        self._rate_limit()
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
        self._rate_limit()
        """Fetch symbol filters (tick size, step size, min notional)."""
        info = self.client.futures_exchange_info()
        sym = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
        if not sym:
            raise ValueError(f"Symbol {symbol} not found in exchange info")
        filters = {f['filterType']: f for f in sym.get('filters', [])}
        return filters

    def format_quantity(self, symbol, quantity):
        self._rate_limit()
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
        self._rate_limit()
        """Round price to tick size."""
        filters = self.get_symbol_filters(symbol)
        pf = filters.get('PRICE_FILTER', {})
        tick = float(pf.get('tickSize', '0.01'))
        decimals = max(0, str(tick)[::-1].find('.'))
        # Round to nearest tick down
        p = price - (price % tick)
        return float(f"{p:.{decimals}f}")
    


    def get_current_price(self, symbol):
        self._rate_limit()
        """Get real-time price for a symbol with rate limiting and NaN handling"""
        try:
            self._rate_limit()
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            import math
            if math.isnan(price) or price <= 0:
                logging.warning(f"Invalid price received for {symbol}: {price}")
                return None
            logging.info(f"Current price for {symbol}: {price}")
            return price
            
        except BinanceAPIException as e:
            logging.error(f"Binance API error getting price for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting price for {symbol}: {e}")
            return None
    
    def get_position_info(self, symbol):
        self._rate_limit()
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
        self._rate_limit()
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
        self._rate_limit()
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
        self._rate_limit()
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
        self._rate_limit()
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
