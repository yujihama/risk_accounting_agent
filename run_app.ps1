# Risk Analysis Agent Streamlit App 起動スクリプト

Write-Host "リスク分析AIエージェントを起動します..." -ForegroundColor Green

# 仮想環境の作成と有効化
if (!(Test-Path "venv")) {
    Write-Host "仮想環境を作成しています..." -ForegroundColor Yellow
    python -m venv venv
}

# 仮想環境を有効化
Write-Host "仮想環境を有効化しています..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# 依存関係のインストール
Write-Host "依存関係をインストールしています..." -ForegroundColor Yellow
pip install -r requirements.txt

# Streamlitアプリケーションの起動
Write-Host "Streamlitアプリケーションを起動しています..." -ForegroundColor Green
python -m streamlit run src/risk_analysis_app.py 