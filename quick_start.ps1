# 簡易起動スクリプト（仮想環境なし）

Write-Host "リスク分析AIエージェントを起動します..." -ForegroundColor Green

# 依存関係のインストール確認
Write-Host "必要なパッケージをインストールしています..." -ForegroundColor Yellow
pip install -r requirements.txt

# Streamlitアプリケーションの起動
Write-Host "Streamlitアプリケーションを起動しています..." -ForegroundColor Green
python -m streamlit run src/risk_analysis_app.py 