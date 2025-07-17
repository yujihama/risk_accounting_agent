import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import io
import base64
from langchain_openai import ChatOpenAI

# --- 長いコードブロック行が画面からはみ出ないようにCSSを注入 ---
if "code_wrap_style_injected" not in st.session_state:
    st.markdown(
        """
        <style>
        /* コードブロックが横にあふれないように折り返し + スクロールを有効化 */
        div[data-testid="stCodeBlock"] pre,
        div[data-testid="stCode"] pre {
            white-space: pre-wrap;      /* 行を折り返す */
            word-break: break-word;     /* 長い単語も折り返す */
            overflow-x: auto;           /* 必要に応じて横スクロール */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.session_state["code_wrap_style_injected"] = True

def show_results_page():
    """結果表示ページ (GraphState と従来 dict の両方に対応)"""
    st.subheader("分析結果")
    st.divider()
    
    # 分析結果が存在するかチェック
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.warning("分析結果がありません。先に分析を実行してください。")
        return
    
    results = st.session_state.analysis_results[-1]
    
    # GraphState または GraphState を .dict() 化した dict を検出
    is_graph_state = (
        (hasattr(results, 'final_assessments') and hasattr(results, 'company_reports')) or
        (isinstance(results, dict) and 'final_assessments' in results and 'company_reports' in results)
    )
    
    if is_graph_state:
        show_graph_state_results(results)


def show_graph_state_results(state):
    """risk_analysis_agent.GraphState 型の結果を表示する"""
    # dict 形式かオブジェクトかで取得方法を分岐
    def _get_attr(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default)) if isinstance(obj, dict) else getattr(obj, key, default)

    companies_input = _get_attr(state, 'all_companies_input', [])
    final_assessments_raw = _get_attr(state, 'final_assessments', [])
    company_reports = _get_attr(state, 'company_reports', {})
    logs = _get_attr(state, 'log', [])

    companies_count = len(companies_input)
    assessments_count = len(final_assessments_raw)
    
    col1, col2 = st.columns(2)
    col1.metric("分析対象企業数", companies_count)
    col2.metric("最終調査件数", assessments_count)
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs(["調査結果一覧", "企業別調査結果レポート", "ログ"])
    
    # --- タブ1: 調査結果一覧 ---
    with tab1:
        if final_assessments_raw:
            # list の要素がオブジェクトか dict かを判定
            def as_dict(a):
                if isinstance(a, dict):
                    return a
                return a.dict() if hasattr(a, 'dict') else {
                    'company_id': getattr(a, 'company_id', ''),
                    'candidate_id': getattr(a, 'candidate_id', ''),
                    'certainty': getattr(a, 'certainty', ''),
                    'reasoning': getattr(a, 'reasoning', '')
                }
            df_assessments = pd.DataFrame([
                {
                    '企業ID': d['company_id'],
                    '調査候補ID': d['candidate_id'],
                    '確度': d['certainty'],
                    '調査結果': d['reasoning']
                } for d in map(as_dict, final_assessments_raw)
            ])
            
            st.dataframe(df_assessments, use_container_width=True)
            
            # 詳細表示
            for a in final_assessments_raw:
                d = as_dict(a)
                with st.expander(f"企業ID: {d['company_id']} / 候補: {d['candidate_id']}"):
                    st.write(f"**確度**: {d['certainty']}")
                    st.write(f"**調査結果**: {d['reasoning']}")
                    docs = d.get('evidence_docs', [])
                    if docs:
                        st.write("**根拠文書**:")
                        for doc in docs:
                            st.write(f"- {doc}")
        else:
            st.info("最終評価結果がありません。")
    
    # --- タブ2: 企業別レポート ---
    with tab2:
        if company_reports:
            for company_id, report in company_reports.items():
                with st.expander(f"企業ID: {company_id} のレポート"):
                    st.markdown(report)
                    # レポートをダウンロード
                    convert_to_pdf(report)
        else:
            st.info("企業別レポートがありません。")
    
    # --- タブ3: 分析ログ ---
    with tab3:
        if logs:
            for log_entry in logs:
                st.code(log_entry)
        else:
            st.info("ログ情報がありません。")

def show_mail_page():
    """メール作成ページ"""
    st.subheader("子会社への問い合わせメール")
    st.divider()

    # --- 1. 分析結果の存在チェック ---
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.warning("分析結果がありません。先に分析を実行してください。")
        return

    results = st.session_state.analysis_results[-1]

    # GraphState もしくは dict で保持されている場合の共通アクセス関数
    def _get_attr(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default)) if isinstance(obj, dict) else getattr(obj, key, default)

    company_reports = _get_attr(results, 'company_reports', {})
    # companies_input = _get_attr(results, 'all_companies_input', [])

    if not company_reports:
        st.warning("企業別レポートがありません。結果生成後に再度お試しください。")
        return

    # --- 2. UI 構築 ---

    # --- 3. LLM によるメール生成 ---
    if 'generated_email' not in st.session_state:
        report_content = "\n".join(company_reports.values())

        prompt = f"""
あなたは本社の内部監査部の担当者です。以下のレポートを基に、子会社担当者宛ての丁寧な日本語メールを作成してください。

# 要求仕様
- 件名、本文の構成で出力してください。
- 件名はレポートの要点が伝わる簡潔なものとしてください。
- 本文では冒頭の挨拶、要点の説明、依頼事項（レポートに基づき追加で提供してほしい情報や資料）、締めの挨拶を含めてください。
- 全体を 500 文字程度に簡潔にまとめてください。

# レポート内容
""" + report_content + """

""" 
        try:
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
            response = llm.invoke(prompt)
            generated_text = response.content if hasattr(response, 'content') else str(response)
            st.session_state.generated_email = generated_text
        except Exception as e:
            st.error(f"メール生成中にエラーが発生しました: {str(e)}")

    # --- 4. 生成結果の表示とダウンロード ---
    if 'generated_email' in st.session_state and st.session_state.generated_email:
        st.text_area("メール文面", value=st.session_state.generated_email, height=400)
        st.download_button(
            label="テキストとしてダウンロード",
            data=st.session_state.generated_email,
            file_name=f"mail.txt",
            mime="text/plain"
        )
    if st.button("メール送信", disabled=not st.session_state.generated_email):
        st.success(f"メールが送信されました（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）")

def convert_to_pdf(markdown_content):
    """MarkdownをPDFに変換"""
    try:
        # 一時的にMarkdownファイルを作成
        import os
        import datetime
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        temp_md_path = f"tmp/report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

    except Exception as e:
        st.error(f"PDF変換中にエラーが発生しました: {str(e)}")
        return None
