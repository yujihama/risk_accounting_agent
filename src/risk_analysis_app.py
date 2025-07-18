import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import PyPDF2
import io

from results_display import show_results_page, show_mail_page

# ページ設定
st.set_page_config(
    page_title="Agentic-AI Subsidiary Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        text-align: left;
        color: #1f77b4;
        margin-bottom: 3rem !important;
        border-bottom: 1px solid #1f77b4;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .analysis-log {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .node-highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #2196f3;
        margin: 0.25rem 0;
    }
    /* プライマリボタンの色を落ち着いたブルーに変更 */
    button[data-testid="baseButton-primary"],
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: #ffffff;
        border: 1px solid #1f77b4;
    }
    button[data-testid="baseButton-primary"]:hover,
    div.stButton > button:first-child:hover {
        background-color: #1665a2;
        color: #ffffff;
        border: 1px solid #1665a2;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
def initialize_session_state():
    """セッション状態を初期化"""
    # 基本的な状態変数を初期化
    default_states = {
        'common_docs': [],
        'company_data': [],
        'analysis_status': 'idle',
        'analysis_logs': [],
        'current_node': None,
        'analysis_results': [],
        'openai_api_key': '',
        'analysis_thread': None,
        'error_message': None,
        'doc_folder_path': os.path.join(os.getcwd(), 'data', 'docs') if os.path.isdir(os.path.join(os.getcwd(), 'data', 'docs')) else '',
        'additional_doc_folder_path': os.path.join(os.getcwd(), 'data', 'additional_data') if os.path.isdir(os.path.join(os.getcwd(), 'data', 'additional_data')) else ''
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """メインアプリケーション"""
    initialize_session_state()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">Agentic-AI Subsidiary Analysis</h1>', unsafe_allow_html=True)
    
    # サイドバーでページ選択
    with st.sidebar:
        
        # ページ選択
        st.subheader("ページ選択")
        page = st.selectbox(
            label="表示ページを選択",
            options=["データ設定", "調査", "結果表示", "メール作成"],
            index=0
        )

        st.divider()

        # OpenAI APIキー設定
        st.subheader("API設定")
        api_key = st.text_input(
            "OpenAI APIキー",
            type="password",
            value=st.session_state.get('openai_api_key', '')
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("APIキーが設定されました")
        elif 'openai_api_key' not in st.session_state:
            st.warning("APIキーを入力してください")
        
        st.divider()

    # 選択されたページを表示
    if page == "データ設定":
        show_data_setup_page()
    elif page == "調査":
        show_analysis_page()
    elif page == "結果表示":
        show_results_page()
    elif page == "メール作成":
        show_mail_page()

def show_data_setup_page():
    """データ設定ページ"""
    st.subheader("データ設定")
    st.divider()
    
    # 状態表示
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('doc_folder_path'):
            st.success("ドキュメントフォルダ: 設定済み")
        else:
            st.warning("ドキュメントフォルダ: 未設定")
    
    with col2:
        if st.session_state.company_data:
            st.success("子会社指標データ: 設定済み")
        else:
            st.warning("子会社指標データ: 未設定")
    
    # 1. ドキュメント
    tab1, tab2 = st.tabs(["ドキュメントフォルダ設定", "子会社指標データアップロード"])
    with tab1:
        folder_path = st.text_input(
            "フォルダパス",
            value=('data/docs'),
        )

        pdf_count = 0
        if st.button("フォルダを設定"):
            # 入力値が絶対パスでなければ、カレントディレクトリと結合
            selected_path = folder_path.strip()
            if not os.path.isabs(selected_path):
                selected_path = os.path.join(os.getcwd(), selected_path)

            if os.path.isdir(selected_path):
                st.session_state.doc_folder_path = selected_path
                pdf_count = len([f for f in os.listdir(selected_path) if f.lower().endswith('.pdf')])
            else:
                st.error("有効なフォルダパスを入力してください。実際のパス: " + selected_path)


    with tab2:
        # データ入力方法選択
        input_method = st.radio(
            "データ入力方法",
            ["JSONファイルアップロード", "CSVファイルアップロード"],
            horizontal=True
        )
        
        if input_method == "JSONファイルアップロード":
            show_json_upload()
        elif input_method == "CSVファイルアップロード":
            show_csv_upload()
            

    # 現在の設定状況を表示
    # if st.session_state.get('doc_folder_path'):
    #     st.info(f"設定済みフォルダ: {folder_path}")
    
    
    # 企業データ表示
    if st.session_state.company_data:
        st.subheader("登録済み子会社指標データ")
        df = pd.DataFrame(st.session_state.company_data)
        # abnormal_indicators 列がリスト/辞書の場合にそのまま表示すると [object Object] となるため、
        # JSON 文字列にシリアライズして可読化する
        if 'abnormal_indicators' in df.columns:
            df['abnormal_indicators'] = df['abnormal_indicators'].apply(
                lambda x: (
                    json.dumps(x, ensure_ascii=False, indent=2)
                    if not isinstance(x, str) else x
                ) if x is not None else ''
            )
        st.dataframe(df, use_container_width=False)
        
        if st.button("子会社指標データをクリア"):
            st.session_state.company_data = []
            st.rerun()
    
    st.divider()
    
    if st.button("子会社指標データ取り込み"):
        load_sample_company_data()


def process_pdf_files(uploaded_files):
    """PDFファイルを処理"""
    # max_size_bytes = int(max_file_size.replace('MB', '')) * 1024 * 1024
    
    processed_docs = []
    
    for uploaded_file in uploaded_files:
        try:
            # PDFからテキストを抽出
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                st.warning(f"ファイル {uploaded_file.name} からテキストを抽出できませんでした。")
                continue
            
            doc_info = {
                'filename': uploaded_file.name,
                'content': text_content,
                'pages': len(pdf_reader.pages),
                'processed_at': datetime.now().isoformat()
            }
            
            processed_docs.append(doc_info)
            
        except Exception as e:
            st.error(f"ファイル {uploaded_file.name} の処理中にエラーが発生しました: {str(e)}")
    
    if processed_docs:
        st.session_state.common_docs.extend(processed_docs)
        st.success(f"{len(processed_docs)}件のドキュメントを処理しました。")
        st.rerun()

def show_json_upload():
    """JSONファイルアップロード"""
    st.write("企業データのJSONファイルをアップロードしてください。")
    
    # JSONフォーマット例を表示
    with st.expander("JSONフォーマット例"):
        sample_json = [
            {
                "company_name": "サンプル企業A",
                "industry": "IT",
                "employees": "500-1000",
                "founded_year": 2010,
                "revenue": "100",
                "location": "東京都",
                "additional_info": "クラウドサービス提供",
                "abnormal_indicators": {
                    "売上高減少率": "5%",
                    "離職率": "10%"
                }
            }
        ]
        st.json(sample_json)
    
    uploaded_json = st.file_uploader("JSONファイルを選択", type=['json'], key="json_uploader")
    
    if uploaded_json:
        try:
            json_data = json.load(uploaded_json)
            
            if isinstance(json_data, list):
                st.session_state.company_data.extend(json_data)
                st.success(f"{len(json_data)}件の企業データを読み込みました。")
                st.rerun()
            else:
                st.error("JSONファイルは配列形式である必要があります。")
        
        except json.JSONDecodeError:
            st.error("無効なJSONファイルです。")
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {str(e)}")

def show_csv_upload():
    """CSVファイルアップロード"""
    st.write("企業データのCSVファイルをアップロードしてください。")
    
    uploaded_csv = st.file_uploader("CSVファイルを選択", type=['csv'], key="csv_uploader")
    
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            
            # データプレビュー
            st.subheader("データプレビュー")
            st.dataframe(df.head())
            
            if st.button("CSVデータを読み込み"):
                company_data = df.to_dict('records')
                st.session_state.company_data.extend(company_data)
                st.success(f"{len(company_data)}件の企業データを読み込みました。")
                st.rerun()
        
        except Exception as e:
            st.error(f"CSVファイル読み込みエラー: {str(e)}")

# def load_sample_documents():
#     """サンプルドキュメントを読み込み"""
#     sample_docs = [
#         {
#             'filename': 'sample_meeting_minutes.pdf',
#             'content': '''会議議事録
# 日時: 2024年6月15日
# 議題: 四半期業績レビュー

# 主な議論点:
# 1. 売上高が前年同期比で5%減少
# 2. 新規顧客獲得が計画を下回る
# 3. システム障害による機会損失
# 4. 競合他社の価格攻勢への対応

# 決定事項:
# - マーケティング戦略の見直し
# - システム安定性の向上
# - 価格競争力の強化''',
#             'pages': 3,
#             'processed_at': datetime.now().isoformat()
#         },
#         {
#             'filename': 'sample_financial_report.pdf',
#             'content': '''財務レポート
# 期間: 2024年第1四半期

# 業績サマリー:
# - 売上高: 95億円（前年同期比-5%）
# - 営業利益: 8億円（前年同期比-15%）
# - 純利益: 5億円（前年同期比-20%）

# リスク要因:
# - 主要顧客の契約更新不確実性
# - 原材料費の高騰
# - 為替変動リスク
# - 人材確保の困難''',
#             'pages': 5,
#             'processed_at': datetime.now().isoformat()
#         }
#     ]
    
#     st.session_state.common_docs.extend(sample_docs)
#     st.success("サンプルドキュメントを読み込みました。")
#     st.rerun()

def load_sample_company_data():
    """サンプル企業データを読み込み"""
    sample_companies = [
        {
            "company_name": "c1012株式会社",
            "additional_info": """
            会社名: 株式会社c1012
            設立: 2015年4月1日
            資本金: 5,000万円
            従業員数: 45名
            事業内容: 精密機械部品の製造・販売
            """,
            "abnormal_indicators": [{
                "検知された指標値推移": "売上債権回転期間の上昇傾向に対して営業キャッシュフローが減少傾向にある",
                "売上債権回転期間": {
                    "2022年6月": "2.7",
                    "2022年9月": "4.3",
                    "2022年12月": "5.7",
                    "2023年3月": "5.6",
                    "2023年6月": "5.9",
                    "2023年9月": "6.1",
                    "2023年12月": "7.8",
                    "2024年3月": "10.4",
                    "2024年6月": "9.9",
                    "2024年9月": "11.0",
                    "2024年12月": "11.6"
                },
                "営業キャッシュフロー": {
                    "2022年6月": "123",
                    "2022年9月": "120",
                    "2022年12月": "125",
                    "2023年3月": "35",
                    "2023年6月": "-118",
                    "2023年9月": "-216",
                    "2023年12月": "-183",
                    "2024年3月": "-197",
                    "2024年6月": "-250",
                    "2024年9月": "-298",
                    "2024年12月": "-365"
                }
            },
            {
                "検知された指標値推移": "有利子負債比率が減少傾向にあり、かつ利息率が上昇傾向にある",
                "有利子負債比率": {
                    "2022年6月": "3.3",
                    "2022年9月": "2.6",
                    "2022年12月": "2.5",
                    "2023年3月": "2.3",
                    "2023年6月": "2.4",
                    "2023年9月": "2.6",
                    "2023年12月": "3.5",
                    "2024年3月": "2.9",
                    "2024年6月": "2.3",
                    "2024年9月": "2.1",
                    "2024年12月": "1.8"
                },
                "利息率": {
                    "2022年6月": "1.0",
                    "2022年9月": "1.2",
                    "2022年12月": "1.2",
                    "2023年3月": "1.3",
                    "2023年6月": "1.3",
                    "2023年9月": "1.2",
                    "2023年12月": "1.2",
                    "2024年3月": "1.3",
                    "2024年6月": "1.6",
                    "2024年9月": "1.6",
                    "2024年12月": "2.1"
                }
            }]
        }
    ]
    
    st.session_state.company_data.extend(sample_companies)
    st.success("サンプル企業データを読み込みました。")
    st.rerun()

def show_analysis_page():
    """分析実行ページ"""
    st.subheader("分析実行")
    st.divider()
    
    # 前提条件チェック
    if not st.session_state.get('doc_folder_path') or not st.session_state.company_data:
        st.error("分析を実行するには、共通ドキュメントフォルダと企業データの両方が必要です。")
        st.write("データ設定ページで必要なデータを設定してください。")
        return
    
    # ------------------------------
    # 分析実行ボタン (活性/非活性制御)
    # ------------------------------
    # 必須条件の不足をチェック
    missing_api_key = ('openai_api_key' not in st.session_state or not st.session_state.openai_api_key)
    missing_company_data = (not st.session_state.company_data)

    if missing_api_key:
        st.error("分析を実行するには、サイドバーでOpenAI APIキーを設定してください。")
    if missing_company_data:
        st.error("分析を実行するには、企業データを入力してください。")

    # 分析結果をもとに再分析
    button_reanalyze = "analysis_results" in st.session_state and len(st.session_state.analysis_results) > 0

    # 分析開始ボタン
    if button_reanalyze:
        company_id = st.selectbox("分析する子会社を選択", options=st.session_state.analysis_results[-1]["company_reports"].keys())
        if st.button("子会社からの提供資料をもとに分析", type="primary", use_container_width=True):
            start_analysis(reanalyze=True, company_id=company_id)
    else:
        if st.button("分析開始", type="primary", use_container_width=True):
            # ダブルクリック等を防ぐため、idle 状態の時のみ開始
            if st.session_state.analysis_status == 'idle':
                start_analysis()

    # 分析をリセット
    if st.button("分析をリセット"):
        st.session_state.analysis_results = []
        st.session_state.analysis_status = 'idle'
        st.session_state.current_node = None
        st.rerun()

def start_analysis(reanalyze=False, company_id=None):
    """分析を開始"""
    try:
        # セッション状態を安全に初期化
        st.session_state.analysis_status = 'running'
        st.session_state.current_node = "初期化中"
        
        # 実際のリスク分析エージェントを実行
        run_real_analysis(reanalyze, company_id)
        
    except Exception as e:
        st.session_state.analysis_status = 'error'
        st.session_state.error_message = str(e)
        st.session_state.analysis_logs.append(f"エラーが発生しました: {str(e)}")
        st.error(f"分析開始中にエラーが発生しました: {str(e)}")
    
def run_real_analysis(reanalyze=False, company_id=None):
    """実際のリスク分析エージェントを実行"""
    try:
        # 環境変数にAPIキーを設定
        import os
        if not st.session_state.get('openai_api_key'):
            raise ValueError("OpenAI APIキーが設定されていません")
        
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
        
        from risk_analysis_agent import CompanyInput, app
        from langchain_core.documents import Document
        
        docs_folder_path = st.session_state.get('doc_folder_path')
        additional_doc_folder_path = os.path.join(os.getcwd(), 'data', 'additional_data')
        if not docs_folder_path:
            raise ValueError("共通ドキュメントフォルダが設定されていません")
        
        # 企業データを CompanyInput に変換
        # company_idが指定されている場合は、その子会社のみを分析対象とする
        companies_input = []
        if company_id is None:
            for i, company_data in enumerate(st.session_state.company_data):
                try:
                    company_input = CompanyInput(
                        company_id=f"company_{i+1}",
                        company_name=company_data.get('company_name', f'企業{i+1}'),
                        summary=company_data.get('additional_info', company_data.get('business_description', '')),
                        key_metrics={},
                        abnormal_indicators=(
                            {f"indicator_{idx+1}": item for idx, item in enumerate(company_data.get('abnormal_indicators', []))}
                            if isinstance(company_data.get('abnormal_indicators', []), list)
                            else company_data.get('abnormal_indicators', {})
                        )
                    )
                    companies_input.append(company_input)
                except Exception as e:
                    st.session_state.analysis_logs.append(f"企業データ変換エラー (企業{i+1}): {str(e)}")
                    continue
        else:
            for i, company_data in enumerate(st.session_state.company_data):
                try:
                    if f"company_{i+1}" == company_id:
                        company_input = CompanyInput(
                            company_id=f"company_{i+1}",
                            company_name=company_data.get('company_name', f'企業{i+1}'),
                            summary=company_data.get('additional_info', company_data.get('business_description', '')),
                            key_metrics={},
                            abnormal_indicators=(
                                {f"indicator_{idx+1}": item for idx, item in enumerate(company_data.get('abnormal_indicators', []))}
                                if isinstance(company_data.get('abnormal_indicators', []), list)
                                else company_data.get('abnormal_indicators', {})
                            )
                        )
                        companies_input.append(company_input)
                        break
                except Exception as e:
                    st.session_state.analysis_logs.append(f"企業データ変換エラー (企業{i+1}): {str(e)}")
                    continue

        if not companies_input:
            raise ValueError("有効な企業データが見つかりません")

        st.session_state.analysis_logs.append(f"{len(companies_input)}件の企業データを変換しました")

        # UI では PDF を読み込まず、エージェント側がフォルダを走査して取り込みを行う。そのため common_qualitative_docs は空リスト。

        initial_input_data = {
            "all_companies_input": companies_input,
            "common_qualitative_docs": [],
            "docs_folder_path": docs_folder_path,
            "additional_doc_folder_path": additional_doc_folder_path,
            "max_depth": 5,
            "reanalyze": reanalyze,
            "company_id": company_id,
            "target_company_reports": st.session_state.analysis_results[-1]["company_reports"].get(company_id, "") if reanalyze else ""
        }

        with st.spinner("リスク分析エージェントを実行しています..."):
            # 実際の分析を実行
            final_state = app.invoke(initial_input_data, config={"recursion_limit": 100})
        
        # 結果を保存
        if reanalyze == False:
            st.session_state.analysis_results = [final_state]
        else:
            st.session_state.analysis_results.append(final_state)

        # エージェントが保持する詳細ログを UI 用ログに統合
        if hasattr(final_state, "log") and isinstance(final_state.log, list):
            # 区切り線を追加
            st.session_state.analysis_logs.append("--- エージェント詳細ログ ---")
            st.session_state.analysis_logs.extend(final_state.log)

        st.session_state.analysis_status = 'completed'
        st.session_state.current_node = "分析完了"
        st.success("分析が正常に完了しました")

    except Exception as e:
        st.session_state.analysis_status = 'error'
        st.session_state.error_message = str(e)
        st.session_state.analysis_logs.append(f"エラーが発生しました: {str(e)}")
        
        # デバッグ情報を追加
        import traceback
        error_details = traceback.format_exc()
        st.session_state.analysis_logs.append(f"詳細エラー情報: {error_details}")
        
        # エラーをユーザーに表示
        st.error(f"分析中にエラーが発生しました: {str(e)}")
        
        # デバッグ用にエラー詳細を表示
        with st.expander("エラー詳細（デバッグ用）"):
            st.code(error_details)

if __name__ == "__main__":
    main()

