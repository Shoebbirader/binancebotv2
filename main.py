import os
import json
import pandas as pd
import numpy as np
import time
from data_collector import DataCollector
from feature_engineering import engineer_features, prepare_training_data_enhanced
from ml_model import (
    CryptoLSTM, 
    SimpleTransformer, 
    train_model_balanced, 
    predict_model
)
from risk_manager import RiskManager
from binance_interface import BinanceInterface
from logger import log_error, log_info, log_warning, log_trade, log_prediction
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def main():
    # Load configuration
    with open('config/trading_config.json') as f:
        config = json.load(f)
    
    # Initialize components
    data_collector = DataCollector('config/trading_config.json')
    risk_manager = RiskManager(config)
    
    # Initialize Binance interface
    try:
        use_testnet = bool(config.get('testnet', False))
        binance = BinanceInterface(use_testnet=use_testnet)
        log_info("Binance API connection established")
        # Show current account balance summary
        try:
            balances = binance.get_account_balance()
            usdt = balances.get('USDT', {})
            wallet = float(usdt.get('wallet_balance', 0))
            available = float(usdt.get('available_balance', 0))
            env_label = 'TESTNET' if use_testnet else 'LIVE'
            print(f"💼 Account [{env_label}] | USDT Wallet: {wallet:.2f} | Available: {available:.2f}")
        except Exception as e:
            log_warning(f"Unable to display account balance: {e}")
    except Exception as e:
        log_error(f"Failed to initialize Binance API: {e}")
        return
    
    # Training parameters
    lookback = config.get('lookback', 40)
    epochs = config.get('epochs', 50)
    learning_rate = config.get('learning_rate', 0.0003)
    batch_size = config.get('batch_size', 32)
    prediction_threshold = config.get('prediction_threshold', 0.6)
    
    print("🚀 Starting BinanceBot with FULL LIVE TRADING enabled!")
    print(f" Trading threshold: {prediction_threshold}")
    print(f"💰 Position size: {config.get('position_size_pct', 10)}% of balance")
    print(f"⚡ Leverage range: {config.get('min_leverage', 3)}x - {config.get('max_leverage', 8)}x")
    print(f"Configuration: Lookback={lookback}, Epochs={epochs}, LR={learning_rate}")
    print("=" * 60)
    
    # Parallel per-symbol processing
    binance_lock = threading.Lock()
    def process_symbol(symbol):
        print(f"--- Processing {symbol} ---")
        try:
            # Collect data
            data = data_collector.fetch_historical(symbol)
            if data is None or len(data) < lookback + 100:
                print(f"Insufficient data for {symbol}, skipping...")
                return symbol, None
            
            print(f"Data loaded: {len(data)} samples for {symbol}")
            
            # Engineer features
            data = engineer_features(data)
            feature_columns = [col for col in data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            print(f"Successfully engineered {len(feature_columns)} features")
            
            # Use enhanced data preparation
            X, y = prepare_training_data_enhanced(data, feature_columns, lookback, target_horizon=5)
            if X.size == 0 or y.size == 0:
                print(f"No training data after preparation for {symbol}, skipping...")
                return symbol, None
            print(f"Training {symbol}, X shape: {X.shape}, y shape: {y.shape}")
            
            # Convert y to int64 for bincount
            y_int = y.astype(np.int64)
            try:
                dist = np.bincount(y_int)
            except Exception:
                dist = []
            print(f"Features: {len(feature_columns)}, Target distribution: {dist}")
            
            # Train LSTM
            print("Training Balanced LSTM...")
            try:
                lstm_model = CryptoLSTM(input_dim=len(feature_columns), hidden_dim=64, output_dim=1)
                lstm_model = train_model_balanced(lstm_model, X, y, epochs=epochs, batch_size=batch_size, lr=learning_rate)
                pred_lstm = predict_model(lstm_model, X[-1:])
                print(f"LSTM prediction: {pred_lstm:.4f}")
            except Exception as e:
                print(f"LSTM training error for {symbol}: {e}")
                log_error(f"LSTM training error for {symbol}: {e}")
                pred_lstm = 0.5
            
            # Train Transformer
            print("Training Balanced Transformer...")
            try:
                transformer_model = SimpleTransformer(
                    input_dim=len(feature_columns),
                    d_model=64,
                    nhead=8,
                    num_layers=2,
                    output_dim=1
                )
                transformer_model = train_model_balanced(transformer_model, X, y, epochs=epochs, batch_size=batch_size, lr=learning_rate)
                pred_transformer = predict_model(transformer_model, X[-1:])
                print(f"Transformer prediction: {pred_transformer:.4f}")
            except Exception as e:
                print(f"Transformer training or prediction error for {symbol}: {e}")
                log_error(f"Transformer: {symbol}: {e}")
                pred_transformer = 0.5
            
            # Ensemble prediction
            ensemble_pred = (pred_lstm + pred_transformer) / 2
            print(f"{symbol} ensemble prediction: {ensemble_pred:.4f}")
            
            # Risk management decision and LIVE TRADE EXECUTION
            if ensemble_pred > prediction_threshold:
                action = "BUY"
                print(f"🟢 {symbol}: Prediction {ensemble_pred:.4f} above threshold {prediction_threshold}, {action} signal!")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)
                with binance_lock:
                    execute_live_trade(symbol, "BUY", ensemble_pred, risk_manager, binance, config)
            elif ensemble_pred < config.get('sell_threshold', 0.4):
                action = "SELL"
                print(f"🔴 {symbol}: Prediction {ensemble_pred:.4f} below sell threshold, {action} signal!")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)
                with binance_lock:
                    execute_live_trade(symbol, "SELL", ensemble_pred, risk_manager, binance, config)
            else:
                action = "HOLD"
                print(f"🟡 {symbol}: Prediction {ensemble_pred:.4f} below threshold {prediction_threshold}, no trade.")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)
            
            return symbol, ensemble_pred
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            log_error(f"Main processing error for {symbol}: {e}")
            return symbol, None

    max_workers = min(len(config['symbols']), max(1, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in config['symbols']}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _sym, _pred = future.result()
            except Exception as e:
                print(f"Unhandled exception in worker for {symbol}: {e}")
                log_error(f"Worker exception for {symbol}: {e}")
    
    print("\n✅ Bot processing complete!")
    
    # Show position summary
    summary = risk_manager.get_position_summary()
    print(f"\n📊 Position Summary:")
    print(f"Active positions: {summary.get('active_positions', 0)}")
    print(f"Total trades: {summary.get('total_positions', 0)}")
    print(f"Total P&L: {summary.get('total_pnl_pct', 0):.2f}%")

def execute_live_trade(symbol, action, confidence, risk_manager, binance, config):
    """Execute LIVE TRADE with full Binance API integration"""
    try:
        # Respect live_trading/auto_execute flags
        if not config.get('live_trading', False) or not config.get('auto_execute', False):
            log_warning(f"Trading disabled by config. Skipping live trade for {symbol} {action}.")
            return

        allowed, reason = risk_manager.can_open_new_position(symbol)
        if not allowed:
            log_warning(f"Entry blocked for {symbol}: {reason}")
            return
        print(f"🚀 EXECUTING LIVE TRADE: {symbol} {action}")
        
        # Get real-time account balance
        try:
            balances = binance.get_account_balance()
            usdt_balance = balances.get('USDT', {}).get('available_balance', 0)
            print(f"💰 Available USDT balance: {usdt_balance:.2f}")
            
            if usdt_balance < 10:  # Minimum balance check
                log_warning(f"Insufficient balance for {symbol}: {usdt_balance} USDT")
                return
                
        except Exception as e:
            log_error(f"Failed to get account balance: {e}")
            # Use fallback balance for testing
            usdt_balance = 1000
            print(f"⚠️ Using fallback balance: {usdt_balance} USDT")
        
        # Get real-time price
        try:
            current_price = binance.get_current_price(symbol)
            print(f"📊 Current price for {symbol}: {current_price}")
        except Exception as e:
            log_error(f"Failed to get price for {symbol}: {e}")
            return
        
        # Check existing position
        try:
            existing_position = binance.get_position_info(symbol)
            if existing_position:
                log_warning(f"Position already exists for {symbol}: {existing_position}")
                return
        except Exception as e:
            log_error(f"Failed to check position for {symbol}: {e}")
            # Continue anyway for testing
        
        # Calculate leverage and position size
        leverage = config.get('min_leverage', 3)  # Use minimum leverage for safety
        position_size = risk_manager.calculate_position_size(usdt_balance, current_price, leverage)
        
        if position_size <= 0:
            log_error(f"Invalid position size calculated: {position_size}")
            return
        
        # Set leverage
        try:
            binance.set_leverage(symbol, leverage)
            print(f"⚡ Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            log_error(f"Failed to set leverage for {symbol}: {e}")
            # Continue anyway for testing
        
        # Calculate stop-loss and take-profit
        stop_loss = risk_manager.calculate_stop_loss_price(current_price, action, config.get('stop_loss_percent', 5))
        take_profit = risk_manager.calculate_take_profit_price(current_price, action, config.get('take_profit_percent', 15))
        
        print(f"🛑 Stop-loss: {stop_loss}")
        print(f" Take-profit: {take_profit}")
        
        # Place the order
        try:
            order_bundle = binance.place_futures_order(
                symbol=symbol,
                side=action,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            main_order = order_bundle.get('main') or {}
            print(f"✅ Order placed successfully: {main_order.get('orderId', 'N/A')}")
            
            # Add to position tracking
            risk_manager.add_position(
                symbol=symbol,
                side=action,
                entry_price=current_price,
                quantity=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                bracket_orders={
                    'main': main_order,
                    'sl': order_bundle.get('sl'),
                    'tp': order_bundle.get('tp')
                }
            )
            
            # Log trade execution
            log_trade(symbol, action, current_price, position_size)
            print(f"💰 LIVE TRADE EXECUTED: {symbol} {action} | Size: {position_size:.6f} | Leverage: {leverage}x")
            print(f"🎯 Trade executed successfully! Monitor position for exit signals.")
            
        except Exception as e:
            log_error(f"Order placement failed for {symbol}: {e}")
            print(f"❌ Order placement failed: {e}")
            return
        
    except Exception as e:
        log_error(f"Live trade execution failed for {symbol}: {e}")
        print(f"❌ Live trade execution failed: {e}")

def monitor_positions(risk_manager, binance, config):
    """Monitor active positions for stop-loss and take-profit"""
    try:
        print("\n🔍 Monitoring active positions...")
        
        for symbol in list(risk_manager.active_positions.keys()):
            try:
                # Get current price
                current_price = binance.get_current_price(symbol)
                
                # Check for exit signals
                exit_signal, position = risk_manager.update_position(symbol, current_price)
                
                if exit_signal:
                    print(f"🚨 {exit_signal} triggered for {symbol} at {current_price}")
                    
                    # Close position
                    close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                    binance.close_position(symbol, close_side, abs(position['quantity']))
                    # Cancel any remaining open SL/TP after closing
                    try:
                        binance.cancel_all_open_orders(symbol)
                    except Exception as e:
                        log_warning(f"Failed to cancel remaining orders for {symbol}: {e}")
                    
                    # Update tracking
                    risk_manager.close_position_tracking(symbol, current_price, exit_signal)
                    
                    print(f"✅ Position closed: {symbol} {exit_signal}")
                
            except Exception as e:
                log_error(f"Error monitoring position {symbol}: {e}")
                continue
                
    except Exception as e:
        log_error(f"Error in position monitoring: {e}")

if __name__ == "__main__":
    main()
