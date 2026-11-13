@echo off
rem 人力Semantle をダブルクリックで起動する（Windows）
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv がありません。README の「セットアップ」を先に実行してください。
    pause
    exit /b 1
)
echo 起動中… ブラウザが自動で開きます（閉じるときはこのウィンドウで Ctrl+C）
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
