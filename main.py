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
from logger import log_error, log_info, log_warning, log_trade, log_prediction, log_paper_trade
from model_manager import ModelManager
from ensemble_predictor import EnhancedEnsemble, ensemble_predictor
from model_performance_tracker import performance_tracker
from position_monitor import PositionMonitor
from data_synchronizer import validate_training_data, synchronize_arrays
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def main():
    # Load and validate configuration
    try:
        with open('config/trading_config.json') as f:
            config = json.load(f)
        
        # Validate configuration
        from config_validator import validate_config, validate_environment_vars
        is_valid, errors = validate_config(config)
        if not is_valid:
            log_error("Configuration validation failed:")
            for error in errors:
                print(f"❌ {error}")
            return
        
        # Validate environment variables
        env_valid, env_errors = validate_environment_vars()
        if not env_valid:
            log_error("Environment validation failed:")
            for error in env_errors:
                print(f"❌ {error}")
            return
            
    except Exception as e:
        log_error(f"Failed to load or validate configuration: {e}")
        return
    
    # Initialize components
    data_collector = DataCollector('config/trading_config.json')
    risk_manager = RiskManager(config)
    adv_risk_mgr = AdvancedRiskManager(max_drawdown_pct=15)
    from market_regime import MarketRegimeDetector
    regime_detector = MarketRegimeDetector()
    model_manager = ModelManager()

    # Initialize trading interface (Paper or Live)
    try:
        if config.get('paper_trading', True):
            # Verify API credentials even for paper trading (needed for real prices)
            import os
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            if not api_key or not api_secret:
                log_error("Binance API credentials required even for paper trading (for real price data)")
                print("❌ Missing Binance API credentials in config/secrets.env")
                return
            
            binance = init_paper_trading(initial_balance=10000.0)
            log_info("Paper trading mode enabled with real price feeds")
            print("📊 Paper trading mode enabled with real price feeds")
            print(f"💰 Paper balance: 10000.00 USDT")
        else:
            use_testnet = bool(config.get('testnet', False))
            binance = BinanceInterface(use_testnet=use_testnet)
            log_info("Live trading mode enabled")
            print("🚨 LIVE TRADING MODE ENABLED - REAL MONEY AT RISK")
            # Show current account balance summary
            try:
                balances = binance.get_account_balance()
                usdt = balances.get('USDT', {})
                wallet = float(usdt.get('wallet_balance', 0))
                available = float(usdt.get('available_balance', 0))
                env_label = 'TESTNET' if use_testnet else 'LIVE'
                print(f"💼 Account [{env_label}] | USDT Wallet: {wallet:.2f} | Available: {available:.2f}")
                
                # Safety check for live trading
                if not use_testnet and available < 50:
                    print("⚠️ WARNING: Low available balance for live trading")
                    
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
    
    # Display correct trading mode
    trading_mode = "PAPER TRADING" if config.get('paper_trading', True) else "LIVE TRADING"
    mode_emoji = "📊" if config.get('paper_trading', True) else "🚨"
    
    print(f"{mode_emoji} Starting BinanceBot in {trading_mode} mode!")
    print(f"🎯 Trading threshold: {prediction_threshold}")
    print(f"💰 Position size: {config.get('position_size_pct', 10)}% of balance")
    print(f"⚡ Leverage range: {config.get('min_leverage', 3)}x - {config.get('max_leverage', 8)}x")
    print(f"📈 Configuration: Lookback={lookback}, Epochs={epochs}, LR={learning_rate}")
    print("=" * 60)
    
    # Portfolio-level price data collection for optimization
    price_data_dict = {}
    # Parallel per-symbol processing
    binance_lock = threading.Lock()
    # --- Collect price data for all symbols first ---
    for symbol in config['symbols']:
        data = data_collector.fetch_historical(symbol)
        if data is not None and len(data) > 0:
            price_data_dict[symbol] = data['close']
    # --- Calculate risk parity weights ---
    rp_weights = None
    if price_data_dict:
        try:
            rp_weights = risk_manager.risk_parity_weights(price_data_dict)
            print(f"Risk parity weights: {rp_weights}")
        except Exception as e:
            print(f"Error calculating risk parity weights: {e}")
            rp_weights = None
    # --- Symbol processing function ---
    def process_symbol(symbol):
        print(f"--- Processing {symbol} ---")
        start_time = time.time()
        try:
            # Collect more data for better training
            data = data_collector.fetch_historical(symbol, limit=1500)
            if data is None or len(data) < lookback + 200:  # Increased minimum data requirement
                print(f"Insufficient data for {symbol}, skipping... (need {lookback + 200}, got {len(data) if data is not None else 0})")
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
            adaptive_config = adapt_config_by_regime(config, regime_summary)

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
            
            # Comprehensive data validation and synchronization
            is_valid, errors = validate_training_data(X, y, symbol)
            if not is_valid:
                print(f"Data validation failed for {symbol}: {errors}")
                # Try to synchronize arrays
                X, y = synchronize_arrays(X, y, symbol)
                if X is None or y is None:
                    print(f"Failed to synchronize data for {symbol}, skipping...")
                    return symbol, None
                
            print(f"Training {symbol}, X shape: {X.shape}, y shape: {y.shape}")

            # Convert y to int64 for bincount
            y_int = y.astype(np.int64)
            try:
                dist = np.bincount(y_int)
            except Exception:
                dist = []
            print(f"Features: {len(feature_columns)}, Target distribution: {dist}")

            # Validate training data
            if len(X) < 100:  # Increased minimum for enhanced models
                print(f"Insufficient training data for {symbol}: {len(X)} samples (need 100+)")
                return symbol, None
            
            # Check class balance and ensure minimum samples per class
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                print(f"Only one class in training data for {symbol}")
                return symbol, None
            
            # Ensure minimum samples per class for reliable training
            min_class_size = min(counts)
            if min_class_size < 20:  # Need at least 20 samples per class
                print(f"Insufficient samples per class for {symbol}: {dict(zip(unique, counts))}")
                return symbol, None
            
            print(f"Training data validation passed: {len(X)} samples, classes: {dict(zip(unique, counts))}")

            # Train Optimized LSTM  
            print("Training Optimized LSTM...")
            try:
                lstm_model = CryptoLSTM(input_dim=len(feature_columns), hidden_dim=64, output_dim=1)  # Balanced hidden_dim
                lstm_model = train_model_balanced(lstm_model, X, y, epochs=epochs, batch_size=batch_size, lr=learning_rate)
                pred_lstm = predict_model(lstm_model, X[-1:])
                print(f"LSTM prediction: {pred_lstm:.4f}")
                
                # Save LSTM model with performance tracking
                model_manager.save_model(lstm_model, symbol, 'lstm', {
                    'features': len(feature_columns),
                    'samples': len(X),
                    'accuracy': 0.0
                })
                
                # Record prediction for performance tracking (confidence not available yet)
                performance_tracker.record_prediction(symbol, 'lstm', pred_lstm, 0.5)
                
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
                
                # Save Random Forest model
                model_manager.save_model(rf_model, symbol, 'random_forest', {
                    'samples': len(X),
                    'features': len(feature_columns)
                })
                
            except Exception as e:
                print(f"Random Forest error for {symbol}: {e}")
                pred_rf = 0.5

            print("Training XGBoost...")
            try:
                xgb_model = train_xgboost(X, y)
                pred_xgb = xgb_model.predict_proba(X[-1:].reshape(1, -1))[0, 1] if xgb_model else 0.5
                print(f"XGBoost prediction: {pred_xgb:.4f}")
                
                if xgb_model:
                    model_manager.save_model(xgb_model, symbol, 'xgboost', {
                        'samples': len(X),
                        'features': len(feature_columns)
                    })
                
            except Exception as e:
                print(f"XGBoost error for {symbol}: {e}")
                pred_xgb = 0.5

            print("Training LightGBM...")
            try:
                lgb_model = train_lightgbm(X, y)
                pred_lgb = lgb_model.predict_proba(X[-1:].reshape(1, -1))[0, 1] if lgb_model else 0.5
                print(f"LightGBM prediction: {pred_lgb:.4f}")
                
                if lgb_model:
                    model_manager.save_model(lgb_model, symbol, 'lightgbm', {
                        'samples': len(X),
                        'features': len(feature_columns)
                    })
                
            except Exception as e:
                print(f"LightGBM error for {symbol}: {e}")
                pred_lgb = 0.5

            # Enhanced ensemble prediction with weighted voting
            predictions_dict = {
                'lstm': pred_lstm,
                'random_forest': pred_rf, 
                'xgboost': pred_xgb,
                'lightgbm': pred_lgb
            }
            
            # Use enhanced ensemble predictor
            ensemble_pred, confidence = ensemble_predictor.weighted_ensemble_predict(predictions_dict)
            
            if ensemble_pred is None or not (0 <= ensemble_pred <= 1):
                print(f"Warning: Invalid ensemble prediction {ensemble_pred} for {symbol}, skipping trade")
                return symbol, None
            
            print(f"📊 {symbol}: Enhanced ensemble prediction {ensemble_pred:.4f} (confidence: {confidence:.3f})")
            
            # Enhanced trading signal generation with balanced thresholds
            action, should_execute = ensemble_predictor.get_trading_signal(
                ensemble_pred, confidence, 
                buy_threshold=0.6,     # Balanced threshold for BUY signals
                sell_threshold=0.4,    # Balanced threshold for SELL signals  
                min_confidence=0.3     # Increased minimum confidence
            )
            
            if not should_execute:
                print(f"🟡 {symbol}: {action} signal but insufficient confidence ({confidence:.3f}), skipping trade.")
                log_prediction(symbol, ensemble_pred, confidence, action)
                return symbol, None

            # Execute the trade based on enhanced ensemble decision
            if action == "BUY":
                print(f"🟢 {symbol}: {action} signal! Prediction: {ensemble_pred:.4f}, Confidence: {confidence:.3f}")
                log_prediction(symbol, ensemble_pred, confidence, action)
                with binance_lock:
                    execute_live_trade(symbol, "BUY", confidence, risk_manager, binance, config, regime_summary, data)
            elif action == "SELL":
                print(f"🔴 {symbol}: {action} signal! Prediction: {ensemble_pred:.4f}, Confidence: {confidence:.3f}")
                log_prediction(symbol, ensemble_pred, confidence, action)
                with binance_lock:
                    execute_live_trade(symbol, "SELL", confidence, risk_manager, binance, config, regime_summary, data)
            # HOLD case already handled above

            elapsed_time = time.time() - start_time
            print(f"✅ Completed {symbol} in {elapsed_time:.1f}s")
            return symbol, ensemble_pred
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Error processing {symbol} after {elapsed_time:.1f}s: {e}")
            log_error(f"Main processing error for {symbol}: {e}")
            return symbol, None

    # Optimized parallel processing with timeout and progress tracking
    max_workers = min(3, max(1, (os.cpu_count() or 2) // 2))  # Reduced workers to prevent timeout
    print(f"🔄 Processing {len(config['symbols'])} symbols with {max_workers} workers...")
    
    completed_symbols = 0
    failed_symbols = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in config['symbols']}
        for future in as_completed(futures, timeout=1800):  # Increased to 30 minutes for enhanced models
            symbol = futures[future]
            try:
                _sym, _pred = future.result(timeout=300)  # Increased to 5 minutes per symbol
                completed_symbols += 1
                print(f"✅ Completed {symbol} ({completed_symbols}/{len(config['symbols'])})")
            except TimeoutError:
                print(f"⏰ Timeout for {symbol} - skipping to next symbol")
                failed_symbols.append(symbol)
                log_error(f"Timeout processing {symbol}")
            except Exception as e:
                print(f"❌ Exception for {symbol}: {e}")
                failed_symbols.append(symbol)
                log_error(f"Worker exception for {symbol}: {e}")
    
    if failed_symbols:
        print(f"⚠️ {len(failed_symbols)} symbols failed: {failed_symbols}")
    print(f"✅ Successfully processed {completed_symbols}/{len(config['symbols'])} symbols")

    # Start position monitoring for paper trading
    if config.get('paper_trading', False):
        position_monitor = PositionMonitor(binance, risk_manager)
        position_monitor.start_monitoring(check_interval=60)  # Check every minute
        print("📊 Position monitoring started for paper trading")

    # Portfolio optimization: correlation and risk parity
    if price_data_dict:
        print("\n📈 Portfolio Correlation Matrix:")
        corr_matrix = risk_manager.compute_correlation_matrix(price_data_dict)
        print(corr_matrix)
        print("\n📊 Risk Parity Weights:")
        rp_weights = risk_manager.risk_parity_weights(price_data_dict)
        print(rp_weights)
        # Store weights for future use
        process_symbol._rp_weights = rp_weights

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

def adapt_config_by_regime(config, regime_summary):
    """Create regime-adaptive configuration without modifying original"""
    adaptive_config = dict(config)
    
    # Validate regime_summary
    volatility = regime_summary.get('volatility', 'NORMAL')
    trend = regime_summary.get('trend', 'SIDEWAYS')
    
    # Adaptive position sizing based on volatility
    if volatility == 'HIGH':
        adaptive_config['position_size_pct'] = max(2, config.get('position_size_pct', 10) // 2)
        adaptive_config['stop_loss_percent'] = min(5, max(1, config.get('stop_loss_percent', 2) * 1.5))
    elif volatility == 'LOW':
        adaptive_config['position_size_pct'] = min(20, config.get('position_size_pct', 10) * 1.5)
        adaptive_config['stop_loss_percent'] = max(0.5, config.get('stop_loss_percent', 2) * 0.8)
    
    # Adaptive take profit based on trend
    if trend == 'STRONG_TREND':
        adaptive_config['take_profit_percent'] = min(15, config.get('take_profit_percent', 5) * 1.8)
    elif trend == 'SIDEWAYS':
        adaptive_config['take_profit_percent'] = max(1.5, config.get('take_profit_percent', 5) * 0.7)
        adaptive_config['position_size_pct'] = max(5, adaptive_config['position_size_pct'] * 0.8)
    
    # Ensure valid ranges
    adaptive_config['position_size_pct'] = max(1, min(50, adaptive_config['position_size_pct']))
    adaptive_config['stop_loss_percent'] = max(0.5, min(10, adaptive_config['stop_loss_percent']))
    adaptive_config['take_profit_percent'] = max(1, min(20, adaptive_config['take_profit_percent']))
    
    return adaptive_config

def execute_live_trade(symbol, action, confidence, risk_manager, binance, config, regime_summary, data):
    """Execute LIVE TRADE with full Binance API integration"""
    try:
        # Respect auto_execute for both live and paper trading
        if not config.get('auto_execute', False):
            log_warning(f"Auto execution disabled by config. Skipping trade for {symbol} {action}.")
            return
        # Safety checks for live trading
        is_live_trading = not config.get('paper_trading', False) and config.get('live_trading', False)
        if is_live_trading:
            # Additional safety checks for live trading
            if confidence < 0.6:  # Higher confidence required for live trades
                log_warning(f"Live trading requires higher confidence. {symbol} {action} confidence: {confidence:.3f} < 0.6")
                return
            
            # Volume check (ensure sufficient volume for execution)
            try:
                # Check 24hr volume to ensure liquidity (only for live trading)
                if hasattr(binance, 'client'):
                    stats = binance.client.futures_24hr_ticker(symbol=symbol)
                    volume_24h = float(stats['volume'])
                    if volume_24h < 100000:  # Minimum volume threshold
                        log_warning(f"Insufficient 24h volume for {symbol}: {volume_24h}")
                        return
                else:
                    log_info(f"Skipping volume check for paper trading: {symbol}")
            except Exception as e:
                log_warning(f"Could not verify volume for {symbol}: {e}")
                return
        
        # Only block trades if both live_trading and paper_trading are false
        if not config.get('live_trading', False) and not config.get('paper_trading', False):
            log_warning(f"All trading disabled by config. Skipping trade for {symbol} {action}.")
            return

        allowed, reason = risk_manager.can_open_new_position(symbol)
        if not allowed:
            log_warning(f"Entry blocked for {symbol}: {reason}")
            return
        print(f"🚀 EXECUTING LIVE TRADE: {symbol} {action}")

        # Get account balance - prioritize paper trading when enabled
        try:
            # Check if we're in paper trading mode
            is_paper_trading = config.get('paper_trading', False)
            balance_before = 0
            
            if is_paper_trading:
                # Get paper trading balance from the paper trading interface
                balances = binance.get_account_balance()
                if isinstance(balances, dict) and 'USDT' in balances:
                    balance_info = balances.get('USDT', {})
                    usdt_balance = float(balance_info.get('available_balance', 10000.0))
                    balance_before = usdt_balance
                else:
                    # Fallback to direct balance access
                    usdt_balance = getattr(binance, 'current_balance', 10000.0)
                    balance_before = usdt_balance
            else:
                # Real Binance API
                balances = binance.get_account_balance()
                if isinstance(balances, dict) and 'USDT' in balances:
                    balance_info = balances.get('USDT', {})
                    usdt_balance = float(balance_info.get('available_balance', 0))
                    balance_before = usdt_balance
                else:
                    usdt_balance = 10000.0
                    balance_before = usdt_balance
            
            print(f"💰 Available USDT balance: {usdt_balance:.2f} ({'Paper Trading' if is_paper_trading else 'Live'})")

            if usdt_balance < 10:  # Minimum balance check
                log_warning(f"Insufficient balance for {symbol}: {usdt_balance} USDT")
                return

        except Exception as e:
            log_error(f"Failed to get account balance: {e}")
            usdt_balance = 10000.0 if config.get('paper_trading', False) else 0.0
            balance_before = usdt_balance
            print(f"⚠️ Using fallback balance: {usdt_balance} USDT")

        # Get real-time price
        try:
            current_price = binance.get_current_price(symbol)
            if current_price is None or np.isnan(current_price) or current_price <= 0:
                log_error(f"Invalid price received for {symbol}: {current_price}")
                return
            print(f"📊 Current price for {symbol}: {current_price}")
        except Exception as e:
            log_error(f"Failed to get price for {symbol}: {e}")
            return

        # Check existing position (with error handling)
        try:
            existing_position = binance.get_position_info(symbol)
            if existing_position:
                log_warning(f"Position already exists for {symbol}: {existing_position}")
                return
        except Exception as e:
            log_warning(f"Could not check existing position for {symbol}: {e}")
            # Continue anyway - position tracking will handle duplicates

        # Use regime-adaptive config (use config passed from process_symbol)
        adaptive_config = config.copy()
        # Apply regime-based adaptations from process_symbol function scope
        adaptive_config = adapt_config_by_regime(adaptive_config, regime_summary)

        # Calculate leverage and position size
        leverage = adaptive_config.get('min_leverage', 3)
        position_size = risk_manager.calculate_position_size(usdt_balance, current_price, leverage)

        if position_size <= 0:
            log_error(f"Invalid position size calculated: {position_size}")
            return

        # Set leverage (if supported by interface)
        try:
            if hasattr(binance, 'set_leverage'):
                binance.set_leverage(symbol, leverage)
                print(f"⚡ Leverage set to {leverage}x for {symbol}")
            else:
                print(f"⚡ Leverage {leverage}x configured for {symbol} (paper trading)")
        except Exception as e:
            log_warning(f"Could not set leverage for {symbol}: {e}")
            # Continue anyway

        # Calculate stop-loss and take-profit
        stop_loss = risk_manager.calculate_stop_loss_price(current_price, action, adaptive_config.get('stop_loss_percent', 5))
        take_profit = risk_manager.calculate_take_profit_price(current_price, action, adaptive_config.get('take_profit_percent', 15))

        # Enhanced validation for stop-loss/take-profit
        def is_valid_stop_loss(p, entry, action):
            if p is None or np.isnan(p) or p <= 0:
                return False
            if action == 'BUY':
                return p < entry  # Stop loss must be below entry for BUY
            elif action == 'SELL':
                return p > entry  # Stop loss must be above entry for SELL
            return False
            
        def is_valid_take_profit(p, entry, action):
            if p is None or np.isnan(p) or p <= 0:
                return False
            if action == 'BUY':
                return p > entry  # Take profit must be above entry for BUY
            elif action == 'SELL':
                return p < entry  # Take profit must be below entry for SELL
            return False
            
        if not is_valid_stop_loss(stop_loss, current_price, action):
            log_warning(f"Invalid stop-loss for {symbol}: {stop_loss}, using fallback.")
            stop_loss_pct = adaptive_config.get('stop_loss_percent', 2) / 100.0
            stop_loss = current_price * (1 - stop_loss_pct if action == 'BUY' else 1 + stop_loss_pct)
            
        if not is_valid_take_profit(take_profit, current_price, action):
            log_warning(f"Invalid take-profit for {symbol}: {take_profit}, using fallback.")
            take_profit_pct = adaptive_config.get('take_profit_percent', 5) / 100.0
            take_profit = current_price * (1 + take_profit_pct if action == 'BUY' else 1 - take_profit_pct)
            
        # Final validation - ensure stop loss and take profit are reasonable
        if action == 'BUY':
            if stop_loss >= current_price:
                stop_loss = current_price * 0.98  # 2% below current price
                log_warning(f"Adjusted stop-loss for BUY {symbol} to {stop_loss}")
            if take_profit <= current_price:
                take_profit = current_price * 1.03  # 3% above current price
                log_warning(f"Adjusted take-profit for BUY {symbol} to {take_profit}")
        else:  # SELL
            if stop_loss <= current_price:
                stop_loss = current_price * 1.02  # 2% above current price
                log_warning(f"Adjusted stop-loss for SELL {symbol} to {stop_loss}")
            if take_profit >= current_price:
                take_profit = current_price * 0.97  # 3% below current price
                log_warning(f"Adjusted take-profit for SELL {symbol} to {take_profit}")

        print(f"🛑 Stop-loss: {stop_loss}")
        print(f" Take-profit: {take_profit}")

        # Advanced order management with regime-based execution
        order_bundle = None
        try:
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

            # Standardized order_bundle handling
            if order_bundle is None or not isinstance(order_bundle, dict) or 'main' not in order_bundle:
                print(f"Order execution failed for {symbol}, skipping position tracking.")
                return
            main_order = order_bundle['main']

        except Exception as e:
            log_error(f"Order execution failed for {symbol}: {e}")
            return

        start_time = time.time()
        end_time = time.time()
        binance.log_latency(start_time, end_time)
        # Slippage analysis (if order_bundle and current_price available)
        if order_bundle is not None and isinstance(order_bundle, dict):
            binance.analyze_slippage(symbol, current_price, order_bundle.get('main', order_bundle))

        # Add to position tracking (with duplicate check)
        order_status = main_order.get('status', 'UNKNOWN')
        if order_status == 'DUPLICATE':
            print(f"⚠️ Duplicate position prevented for {symbol}")
        else:
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
        
        # Log comprehensive trade data to trades.csv
        try:
            # Get updated balance after trade (for paper trading, get from paper trader)
            if is_paper_trading:
                # Get actual paper trading balance
                paper_balance_info = binance.get_account_balance()
                if isinstance(paper_balance_info, dict) and 'USDT' in paper_balance_info:
                    balance_after = float(paper_balance_info['USDT'].get('available_balance', balance_before))
                else:
                    balance_after = binance.current_balance  # Direct access to paper trader balance
            else:
                # Real trading balance
                balances_after = binance.get_account_balance()
                if isinstance(balances_after, dict) and 'USDT' in balances_after:
                    balance_info = balances_after.get('USDT', {})
                    balance_after = float(balance_info.get('available_balance', balance_before))
                else:
                    balance_after = balance_before
            
            # Calculate P&L (for paper trading, this should be 0 for opening trades)
            if is_paper_trading:
                # Opening a position shouldn't show immediate P&L
                pnl = 0.0
                pnl_pct = 0.0
            else:
                pnl = balance_after - balance_before
                pnl_pct = (pnl / balance_before * 100) if balance_before > 0 else 0
            
            log_paper_trade(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=position_size,
                total_value=current_price * position_size,
                balance_before=balance_before,
                balance_after=balance_after,
                position_size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                prediction=confidence,
                confidence=confidence,
                regime=regime_summary,
                pnl=pnl,
                pnl_pct=pnl_pct,
                trade_type='paper' if config.get('paper_trading', False) else 'live',
                status='executed',
                order_id=str(main_order.get('orderId', ''))
            )
            
            print(f"📊 Trade logged to trades.csv for {symbol}")
            
        except Exception as e:
            log_error(f"Error logging trade to CSV: {e}")
        
        trade_type = "PAPER" if config.get('paper_trading', False) else "LIVE"
        print(f"💰 {trade_type} TRADE EXECUTED: {symbol} {action} | Size: {position_size:.6f} | Leverage: {leverage}x")
        print(f"🎯 {trade_type} trade executed successfully! Monitor position for exit signals.")

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
                    
                    # Close position (with error handling)
                    try:
                        close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                        if hasattr(binance, 'close_position'):
                            binance.close_position(symbol, close_side, abs(position['quantity']))
                        else:
                            # For paper trading, close position directly
                            binance.close_position(symbol, close_side, abs(position['quantity']))
                        
                        # Cancel any remaining open orders if method exists
                        if hasattr(binance, 'cancel_all_open_orders'):
                            try:
                                binance.cancel_all_open_orders(symbol)
                            except Exception as e:
                                log_warning(f"Failed to cancel remaining orders for {symbol}: {e}")
                        
                        # Update tracking
                        risk_manager.close_position_tracking(symbol, current_price, exit_signal)
                        
                    except Exception as e:
                        log_error(f"Failed to close position for {symbol}: {e}")
                    
                    print(f"✅ Position closed: {symbol} {exit_signal}")
                
            except Exception as e:
                log_error(f"Error monitoring position {symbol}: {e}")
                continue
                
    except Exception as e:
        log_error(f"Error in position monitoring: {e}")

def run_paper_trading_cycle(paper_trader, risk_manager, config):
    """
    DEPRECATED: This function is not used in current implementation.
    Paper trading is handled directly in the main trading loop.
    """
    log_warning("run_paper_trading_cycle is deprecated - paper trading handled in main loop")
    return

if __name__ == "__main__":
    main()
