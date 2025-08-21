import os
import json
import pandas as pd
import numpy as np
import time
from data_collector import DataCollector
from feature_engineering import engineer_features, prepare_training_data_enhanced
from ml_model import (
    CryptoLSTM, 
    EnhancedTransformer, 
    train_model_balanced, 
    predict_model,
    train_random_forest,
    train_xgboost,
    train_lightgbm
)
from risk_manager import RiskManager
from advanced_risk import AdvancedRiskManager
from binance_interface import BinanceInterface
from paper_trading import PaperTradingInterface, init_paper_trading
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
    adv_risk_mgr = AdvancedRiskManager(max_drawdown_pct=15)
    from market_regime import MarketRegimeDetector
    regime_detector = MarketRegimeDetector()

    # Initialize trading interface (Paper or Live)
    try:
        if config.get('paper_trading', True):
            binance = init_paper_trading(initial_balance=10000.0)
            log_info("Paper trading mode enabled")
            print("📊 Paper trading mode enabled")
            print(f"💰 Paper balance: 10000.00 USDT")
        else:
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
        log_error(f"Failed to initialize trading interface: {e}")
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
    
    # Portfolio-level price data collection for optimization
    price_data_dict = {}
    rp_weights = None  # Ensure rp_weights is always defined
    # Parallel per-symbol processing
    binance_lock = threading.Lock()
    def process_symbol(symbol, rp_weights=None):
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

            # Collect close price series for portfolio optimization
            price_data_dict[symbol] = data['close']

            # Market regime detection
            regime_summary = regime_detector.get_regime_summary(data, feature_columns)
            print(f"Market regime for {symbol}: {regime_summary}")

            # Regime-adaptive parameters
            adaptive_config = dict(config)
            # Volatility regime
            if regime_summary['volatility'] == 'HIGH':
                adaptive_config['position_size_pct'] = max(5, config.get('position_size_pct', 10) // 2)
                adaptive_config['stop_loss_percent'] = min(4, config.get('stop_loss_percent', 2) * 2)
            elif regime_summary['volatility'] == 'LOW':
                adaptive_config['position_size_pct'] = min(15, config.get('position_size_pct', 10) * 2)
                adaptive_config['stop_loss_percent'] = max(1, config.get('stop_loss_percent', 2) // 2)
            # Trend regime
            if regime_summary['trend'] == 'STRONG_TREND':
                adaptive_config['take_profit_percent'] = min(10, config.get('take_profit_percent', 5) * 2)
            elif regime_summary['trend'] == 'SIDEWAYS':
                adaptive_config['take_profit_percent'] = max(2, config.get('take_profit_percent', 5) // 2)
                adaptive_config['position_size_pct'] = max(2, config.get('position_size_pct', 10) // 2)

            # Portfolio risk parity adjustment
            if rp_weights and symbol in rp_weights:
                base_pct = adaptive_config.get('position_size_pct', 10)
                adaptive_config['position_size_pct'] = max(1, int(base_pct * rp_weights[symbol]))
                print(f"[Portfolio] {symbol} position size pct set by risk parity: {adaptive_config['position_size_pct']}")

            # Kelly criterion (optional, demo)
            # If you have win_rate and win_loss_ratio, use:
            # kelly_frac = risk_manager.kelly_criterion(win_rate, win_loss_ratio)
            # adaptive_config['position_size_pct'] = max(1, int(base_pct * kelly_frac))

            # Use enhanced data preparation
            X, y = prepare_training_data_enhanced(data, feature_columns, lookback, target_horizon=5)
            if X is None or y is None or X.size == 0 or y.size == 0:
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

            # Train ensemble models
            print("Training Random Forest...")
            try:
                rf_model = train_random_forest(X, y)
                pred_rf = rf_model.predict_proba(X[-1:].reshape(1, -1))[0, 1]
                print(f"Random Forest prediction: {pred_rf:.4f}")
            except Exception as e:
                print(f"Random Forest error for {symbol}: {e}")
                pred_rf = 0.5

            print("Training XGBoost...")
            try:
                xgb_model = train_xgboost(X, y)
                pred_xgb = xgb_model.predict_proba(X[-1:].reshape(1, -1))[0, 1] if xgb_model else 0.5
                print(f"XGBoost prediction: {pred_xgb:.4f}")
            except Exception as e:
                print(f"XGBoost error for {symbol}: {e}")
                pred_xgb = 0.5

            print("Training LightGBM...")
            try:
                lgb_model = train_lightgbm(X, y)
                pred_lgb = lgb_model.predict_proba(X[-1:].reshape(1, -1))[0, 1] if lgb_model else 0.5
                print(f"LightGBM prediction: {pred_lgb:.4f}")
            except Exception as e:
                print(f"LightGBM error for {symbol}: {e}")
                pred_lgb = 0.5

            # Ensemble prediction
            preds = [pred_lstm, pred_rf, pred_xgb, pred_lgb]
            preds = [p for p in preds if p is not None and not np.isnan(p)]
            if len(preds) == 0:
                ensemble_pred = 0.5
            else:
                ensemble_pred = np.mean(preds)
            print(f"{symbol} ensemble prediction: {ensemble_pred:.4f}")

            # Robust threshold logic: buy > threshold, sell < sell_threshold, hold otherwise
            sell_threshold = config.get('sell_threshold', 0.30)  # Default lower than buy threshold
            if ensemble_pred >= prediction_threshold:
                action = "BUY"
                print(f"🟢 {symbol}: Prediction {ensemble_pred:.4f} >= threshold {prediction_threshold}, {action} signal!")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)
                with binance_lock:
                    execute_live_trade(symbol, "BUY", ensemble_pred, risk_manager, binance, config, regime_summary, data)
            elif ensemble_pred <= sell_threshold:
                action = "SELL"
                print(f"🔴 {symbol}: Prediction {ensemble_pred:.4f} <= sell threshold {sell_threshold}, {action} signal!")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)
                with binance_lock:
                    execute_live_trade(symbol, "SELL", ensemble_pred, risk_manager, binance, config, regime_summary, data)
            else:
                action = "HOLD"
                print(f"🟡 {symbol}: Prediction {ensemble_pred:.4f} between thresholds, no trade.")
                log_prediction(symbol, ensemble_pred, ensemble_pred, action)

            return symbol, ensemble_pred
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            log_error(f"Main processing error for {symbol}: {e}")
            return symbol, None

    max_workers = min(len(config['symbols']), max(1, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, symbol, rp_weights): symbol for symbol in config['symbols']}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _sym, _pred = future.result()
            except Exception as e:
                print(f"Unhandled exception in worker for {symbol}: {e}")
                log_error(f"Worker exception for {symbol}: {e}")

    # Portfolio optimization: correlation and risk parity
    rp_weights = None
    if price_data_dict:
        print("\n📈 Portfolio Correlation Matrix:")
        corr_matrix = risk_manager.compute_correlation_matrix(price_data_dict)
        print(corr_matrix)
        print("\n📊 Risk Parity Weights:")
        rp_weights = risk_manager.risk_parity_weights(price_data_dict)
        print(rp_weights)

    print("\n✅ Bot processing complete!")

    # Show position summary
    summary = risk_manager.get_position_summary()
    
    # Initialize wallet with fallback value
    wallet = 10000.0  # Default paper trading balance
    
    # Try to get actual wallet balance
    try:
        balances = binance.get_account_balance()
        usdt = balances.get('USDT', {})
        wallet = float(usdt.get('wallet_balance', 10000.0))
    except Exception as e:
        print(f"Warning: Could not fetch wallet balance, using default: {e}")
    
    print(f"\n📊 Position Summary:")
    print(f"Active positions: {summary.get('active_positions', 0)}")
    print(f"Total trades: {summary.get('total_positions', 0)}")
    print(f"Total Portfolio P&L: {summary.get('total_pnl_pct', 0):.2f}% | Wallet Balance: {wallet:.2f} USDT")

    # After each trading cycle, update equity and check drawdown
    # Use the wallet balance we already have
    try:
        adv_risk_mgr.update_equity(wallet)
        if adv_risk_mgr.check_drawdown_limit():
            print("Risk limit breached! Consider reducing positions or pausing trading.")
    except Exception as e:
        print(f"Error updating advanced risk manager: {e}")
    
    # Portfolio returns for VaR/ES/Monte Carlo
    try:
        # Example: use daily P&L from position history
        pnl_list = [pos.get('pnl_pct', 0) for pos in risk_manager.position_history]
        returns = np.array(pnl_list)
        if len(returns) > 10:
            adv_risk_mgr.value_at_risk(returns)
            adv_risk_mgr.expected_shortfall(returns)
            adv_risk_mgr.monte_carlo_simulation(returns)
    except Exception as e:
        print(f"Error in advanced risk analytics: {e}")

def execute_live_trade(symbol, action, confidence, risk_manager, binance, config, regime_summary, data):
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

        # Get account balance (paper trading or real)
        try:
            balances = binance.get_account_balance()
            if isinstance(balances, dict) and 'USDT' in balances:
                # Real Binance API
                usdt_balance = balances.get('USDT', {}).get('available_balance', 0)
            else:
                # Paper trading interface
                usdt_balance = balances.get('USDT', {}).get('available_balance', 10000.0)
            
            print(f"💰 Available USDT balance: {usdt_balance:.2f}")

            if usdt_balance < 10:  # Minimum balance check
                log_warning(f"Insufficient balance for {symbol}: {usdt_balance} USDT")
                return

        except Exception as e:
            log_error(f"Failed to get account balance: {e}")
            # Use paper trading balance as fallback
            usdt_balance = 10000.0
            print(f"⚠️ Using paper trading balance: {usdt_balance} USDT")

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

        # Use regime-adaptive config if available
        adaptive_config = config.copy()
        if 'adaptive_config' in locals():
            adaptive_config = adaptive_config

        # Calculate leverage and position size
        leverage = adaptive_config.get('min_leverage', 3)
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

        stop_loss = risk_manager.calculate_stop_loss_price(current_price, action, adaptive_config.get('stop_loss_percent', 5))
        take_profit = risk_manager.calculate_take_profit_price(current_price, action, adaptive_config.get('take_profit_percent', 15))

        # Validate stop-loss/take-profit
        def is_valid_price(p, entry):
            return p is not None and not np.isnan(p) and p != entry and p > 0
        if not is_valid_price(stop_loss, current_price):
            log_warning(f"Invalid stop-loss for {symbol}, using fallback.")
            stop_loss = current_price * (0.98 if action == 'BUY' else 1.02)
        if not is_valid_price(take_profit, current_price):
            log_warning(f"Invalid take-profit for {symbol}, using fallback.")
            take_profit = current_price * (1.02 if action == 'BUY' else 0.98)

        print(f"🛑 Stop-loss: {stop_loss}")
        print(f" Take-profit: {take_profit}")

        # Advanced order management with regime-based execution
        order_bundle = None
        start_time = time.time()
        # Use VWAP for high volume regime
        if regime_summary.get('volume', 'NORMAL') == 'HIGH':
            order_bundle = binance.execute_vwap(symbol, action, position_size, data, order_type='MARKET')
            print(f"VWAP execution used for {symbol}")
        # Use TWAP for low volatility regime
        elif regime_summary.get('volatility', 'NORMAL') == 'LOW':
            order_bundle = binance.execute_twap(symbol, action, position_size, intervals=3, interval_sec=5, order_type='MARKET')
            print(f"TWAP execution used for {symbol}")
        # Use smart order routing for strong trend
        elif regime_summary.get('trend', 'NORMAL') == 'STRONG_TREND':
            order_bundle = binance.smart_order_routing(symbol, action, position_size, order_type='MARKET', price=current_price)
            print(f"Smart order routing used for {symbol}")
        # Use trailing stop for strong take profit config
        elif adaptive_config.get('take_profit_percent', 5) >= 8:
            activation_price = current_price * 1.01 if action == 'BUY' else current_price * 0.99
            callback_rate = 1.0
            order_bundle = binance.place_trailing_stop_order(symbol, action, position_size, activation_price, callback_rate)
            print(f"Trailing stop order used for {symbol}")
        # Use OCO for sideways regime
        elif regime_summary.get('trend', 'NORMAL') == 'SIDEWAYS' or adaptive_config.get('take_profit_percent', 5) <= 3:
            stop_limit_price = stop_loss
            order_bundle = binance.place_oco_order(symbol, action, position_size, take_profit, stop_loss, stop_limit_price)
            print(f"OCO order used for {symbol}")
        # Use scaled orders for large positions
        elif position_size > 2:
            price_levels = [(current_price * (1 + 0.002 * i), position_size / 3) for i in range(3)]
            order_bundle = binance.place_scaled_orders(symbol, action, position_size, price_levels)
            print(f"Scaled orders used for {symbol}")
        else:
            order_bundle = binance.place_futures_order(
                symbol=symbol,
                side=action,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            print(f"Regular order used for {symbol}")

        end_time = time.time()
        binance.log_latency(start_time, end_time)
        # Slippage analysis (if order_bundle and current_price available)
        if order_bundle is not None and isinstance(order_bundle, dict):
            binance.analyze_slippage(symbol, current_price, order_bundle.get('main', order_bundle))

        # Add to position tracking
        main_order = order_bundle if isinstance(order_bundle, dict) else {'orderId': 'N/A'}
        if isinstance(order_bundle, dict) and 'main' in order_bundle:
            main_order = order_bundle['main']
        print(f"✅ Order placed successfully: {main_order.get('orderId', 'N/A')}")
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
                'sl': order_bundle.get('sl') if isinstance(order_bundle, dict) else None,
                'tp': order_bundle.get('tp') if isinstance(order_bundle, dict) else None
            }
        )
        log_trade(symbol, action, current_price, position_size)
        print(f"💰 LIVE TRADE EXECUTED: {symbol} {action} | Size: {position_size:.6f} | Leverage: {leverage}x")
        print(f"🎯 Trade executed successfully! Monitor position for exit signals.")

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

def run_paper_trading_cycle(paper_trader, risk_manager, config):
    """
    Run a single paper trading cycle
    """
    symbols = config['symbols']
    prediction_threshold = config['prediction_threshold']
    sell_threshold = config['sell_threshold']
    
    for symbol in symbols:
        try:
            # Get current price for paper trading
            current_price = paper_trader.get_current_price(symbol)
            if current_price is None:
                # Simulate price if not available
                current_price = 100.0  # Default simulated price
            
            # Update paper trading prices
            paper_trader.update_prices({symbol: current_price})
            
            # Get features and make prediction (simulate with random data for demo)
            # In real implementation, use actual market data
            features = np.random.randn(1, 50)  # Simulated features
            
            # Simulate ensemble prediction
            ensemble_pred = np.random.uniform(0, 1)
            
            print(f"📊 {symbol}: Paper prediction {ensemble_pred:.4f}")
            
            # Generate trading signal
            if ensemble_pred > prediction_threshold:
                action = "BUY"
                side = "LONG"
            elif ensemble_pred < sell_threshold:
                action = "SELL"
                side = "SHORT"
            else:
                continue
            
            # Check if we should enter position
            if risk_manager.should_enter(ensemble_pred):
                position_size = risk_manager.calculate_position_size(
                    symbol, current_price, paper_trader.get_account_balance()
                )
                
                if position_size > 0 and risk_manager.can_open_new_position(symbol):
                    # Calculate stop loss and take profit
                    stop_loss = risk_manager.calculate_stop_loss_price(
                        current_price, side
                    )
                    take_profit = risk_manager.calculate_take_profit_price(
                        current_price, side
                    )
                    
                    # Execute paper trade
                    result = paper_trader.place_futures_order(
                        symbol=symbol,
                        side="BUY" if action == "BUY" else "SELL",
                        quantity=position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    # Track position
                    risk_manager.add_position(symbol, {
                        'side': side,
                        'quantity': position_size,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                    
                    print(f"📝 Paper trade: {action} {position_size} {symbol} @ ${current_price:.4f}")
                    
                    # Monitor and close positions if needed
                    positions = paper_trader.get_account_info()['positions']
                    for pos in positions:
                        if pos['symbol'] == symbol:
                            # Check exit conditions
                            if risk_manager.should_stop_loss(current_price, pos['entry_price'], side):
                                paper_trader.close_position(symbol, "SELL" if side == "LONG" else "BUY", pos['quantity'])
                                print(f"🛑 Paper stop loss: Close {symbol}")
                            elif risk_manager.should_take_profit(current_price, pos['entry_price'], side):
                                paper_trader.close_position(symbol, "SELL" if side == "LONG" else "BUY", pos['quantity'])
                                print(f"💰 Paper take profit: Close {symbol}")
            
        except Exception as e:
            log_error(f"Error in paper trading cycle for {symbol}: {e}")
            continue
    
    # Print paper trading summary
    try:
        summary = paper_trader.get_account_info()
        print(f"💰 Paper Balance: ${summary['balance']:.2f}")
        print(f"📈 Total P&L: ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']:.2f}%)")
        print(f"📊 Active Positions: {summary['active_positions']}")
        print("-" * 50)
    except Exception as e:
        log_error(f"Error getting paper trading summary: {e}")

if __name__ == "__main__":
    main()
