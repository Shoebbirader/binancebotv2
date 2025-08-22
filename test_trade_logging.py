#!/usr/bin/env python3
"""
Test script to verify trade logging functionality for paper trading.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger

def test_trade_logging():
    """Test the trade logging system with paper trading data."""
    
    print("🧪 Testing Trade Logging System...")
    
    # Initialize trade logger
    trade_logger = TradeLogger()
    
    # Test logging a paper trade
    print("📊 Logging test paper trade...")
    trade_logger.log_trade(
        symbol='BTCUSDT',
        action='BUY',
        price=50000.0,
        quantity=0.1,
        total_value=5000.0,
        balance_before=10000.0,
        balance_after=5000.0,
        position_size=0.1,
        leverage=2.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        prediction=0.85,
        confidence=0.85,
        regime='bullish',
        pnl=-0.5,
        pnl_pct=-0.005,
        trade_type='paper',
        status='executed',
        order_id='TEST123456'
    )
    
    # Test logging another trade
    trade_logger.log_trade(
        symbol='ETHUSDT',
        action='SELL',
        price=3000.0,
        quantity=1.5,
        total_value=4500.0,
        balance_before=5000.0,
        balance_after=9500.0,
        position_size=1.5,
        leverage=1.0,
        stop_loss=3100.0,
        take_profit=2900.0,
        prediction=0.75,
        confidence=0.75,
        regime='bearish',
        pnl=100.0,
        pnl_pct=2.0,
        trade_type='paper',
        status='executed',
        order_id='TEST789012'
    )
    
    # Check if trades.csv exists and display contents
    if os.path.exists('trades.csv'):
        print("\n📁 trades.csv found! Contents:")
        df = pd.read_csv('trades.csv')
        print(f"Total trades logged: {len(df)}")
        print("\nRecent trades:")
        print(df[['timestamp', 'symbol', 'action', 'price', 'quantity', 'trade_type', 'pnl', 'balance_after']].tail())
        
        # Verify paper trading data
        paper_trades = df[df['trade_type'] == 'paper']
        print(f"\n📊 Paper trading trades: {len(paper_trades)}")
        
        if len(paper_trades) > 0:
            total_pnl = paper_trades['pnl'].sum()
            print(f"💰 Total P&L from paper trades: ${total_pnl:.2f}")
            
    else:
        print("❌ trades.csv not found!")
    
    # Test portfolio summary
    print("\n📈 Portfolio Summary:")
    summary = trade_logger.get_portfolio_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ Trade logging test completed!")

if __name__ == "__main__":
    test_trade_logging()