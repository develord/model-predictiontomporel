@echo off
title Crypto Trading Bot
cd /d C:\Users\moham\Desktop\crypto\crypto_v10_multi_tf\LIVE_TRADING

:loop
echo [%date% %time%] Starting trading bot...
python main.py
echo [%date% %time%] Bot stopped. Restarting in 30 seconds...
timeout /t 30
goto loop
