import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import io
import base64

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
    
    results = st.session_state.analysis_results
    
    # GraphState または GraphState を .dict() 化した dict を検出
    is_graph_state = (
        (hasattr(results, 'final_assessments') and hasattr(results, 'company_reports')) or
        (isinstance(results, dict) and 'final_assessments' in results and 'company_reports' in results)
    )
    
    if is_graph_state:
        show_graph_state_results(results)
    # else:
    #     # 旧バージョン (サンプル結果など) に対応
    #     # タブで結果を整理
    #     tab1, tab2, tab3, tab4 = st.tabs(["サマリー", "詳細分析", "可視化", "レポート"])
        
    #     with tab1:
    #         show_summary_tab(results)
        
    #     with tab2:
    #         show_detailed_analysis_tab(results)
        
    #     with tab3:
    #         show_visualization_tab(results)
        
    #     with tab4:
    #         show_report_tab(results)

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
    col2.metric("最終リスク評価件数", assessments_count)
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs(["リスク評価", "企業別レポート", "ログ"])
    
    # --- タブ1: リスク評価一覧 ---
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
                    'リスク候補ID': d['candidate_id'],
                    '確実性': d['certainty'],
                    '評価理由': d['reasoning']
                } for d in map(as_dict, final_assessments_raw)
            ])
            
            st.dataframe(df_assessments, use_container_width=True)
            
            # 詳細表示
            for a in final_assessments_raw:
                d = as_dict(a)
                with st.expander(f"企業ID: {d['company_id']} / 候補: {d['candidate_id']}"):
                    st.write(f"**確実性**: {d['certainty']}")
                    st.write(f"**評価理由**: {d['reasoning']}")
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
        else:
            st.info("企業別レポートがありません。")
    
    # --- タブ3: 分析ログ ---
    with tab3:
        if logs:
            for log_entry in logs:
                st.code(log_entry)
        else:
            st.info("ログ情報がありません。")

def show_summary_tab(results):
    """サマリータブ"""
    st.subheader("リスク評価サマリー")
    
    # 全体的なリスクスコア
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_risk = results.get('overall_risk_score', 0)
        st.metric(
            label="総合リスクスコア",
            value=f"{overall_risk:.1f}/10",
            delta=None
        )
    
    with col2:
        risk_level = get_risk_level(overall_risk)
        st.metric(
            label="リスクレベル",
            value=risk_level,
            delta=None
        )
    
    with col3:
        analyzed_companies = len(results.get('company_analyses', []))
        st.metric(
            label="分析対象企業数",
            value=analyzed_companies,
            delta=None
        )
    
    # リスクカテゴリ別スコア
    st.subheader("カテゴリ別リスクスコア")
    
    risk_categories = results.get('risk_categories', {})
    if risk_categories:
        df_categories = pd.DataFrame([
            {'カテゴリ': category, 'スコア': score}
            for category, score in risk_categories.items()
        ])
        
        fig = px.bar(
            df_categories,
            x='カテゴリ',
            y='スコア',
            title="カテゴリ別リスクスコア",
            color='スコア',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 主要な発見事項
    st.subheader("主要な発見事項")
    key_findings = results.get('key_findings', [])
    if key_findings:
        for i, finding in enumerate(key_findings, 1):
            st.write(f"{i}. {finding}")
    else:
        st.info("主要な発見事項はありません。")

def show_detailed_analysis_tab(results):
    """詳細分析タブ"""
    st.subheader("詳細分析結果")
    
    company_analyses = results.get('company_analyses', [])
    
    if not company_analyses:
        st.info("企業別の詳細分析結果はありません。")
        return
    
    # 企業選択
    company_names = [analysis.get('company_name', f'企業{i+1}') 
                    for i, analysis in enumerate(company_analyses)]
    selected_company = st.selectbox("企業を選択", company_names)
    
    if selected_company:
        company_index = company_names.index(selected_company)
        analysis = company_analyses[company_index]
        
        # 企業情報
        st.subheader(f"{selected_company} の分析結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**基本情報**")
            company_info = analysis.get('company_info', {})
            for key, value in company_info.items():
                st.write(f"- {key}: {value}")
        
        with col2:
            st.write("**リスクスコア**")
            risk_score = analysis.get('risk_score', 0)
            st.metric("企業リスクスコア", f"{risk_score:.1f}/10")
        
        # リスク要因
        st.subheader("特定されたリスク要因")
        risk_factors = analysis.get('risk_factors', [])
        if risk_factors:
            for factor in risk_factors:
                with st.expander(f"リスク: {factor.get('type', '不明')}"):
                    st.write(f"**重要度**: {factor.get('severity', 'N/A')}")
                    st.write(f"**説明**: {factor.get('description', 'N/A')}")
                    st.write(f"**推奨対策**: {factor.get('mitigation', 'N/A')}")
        else:
            st.info("特定されたリスク要因はありません。")
        
        # 分析ログ
        st.subheader("分析ログ")
        analysis_logs = analysis.get('analysis_logs', [])
        if analysis_logs:
            for log in analysis_logs:
                st.text(log)
        else:
            st.info("分析ログはありません。")

def show_visualization_tab(results):
    """可視化タブ"""
    st.subheader("分析結果の可視化")
    
    company_analyses = results.get('company_analyses', [])
    
    if not company_analyses:
        st.info("可視化するデータがありません。")
        return
    
    # 企業別リスクスコア比較
    st.subheader("企業別リスクスコア比較")
    
    df_companies = pd.DataFrame([
        {
            '企業名': analysis.get('company_name', f'企業{i+1}'),
            'リスクスコア': analysis.get('risk_score', 0),
            '業界': analysis.get('company_info', {}).get('industry', '不明')
        }
        for i, analysis in enumerate(company_analyses)
    ])
    
    fig = px.bar(
        df_companies,
        x='企業名',
        y='リスクスコア',
        color='業界',
        title="企業別リスクスコア比較"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # リスク分布
    st.subheader("リスクスコア分布")
    
    fig = px.histogram(
        df_companies,
        x='リスクスコア',
        nbins=10,
        title="リスクスコア分布"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 業界別平均リスク
    if len(df_companies['業界'].unique()) > 1:
        st.subheader("業界別平均リスクスコア")
        
        industry_avg = df_companies.groupby('業界')['リスクスコア'].mean().reset_index()
        
        fig = px.pie(
            industry_avg,
            values='リスクスコア',
            names='業界',
            title="業界別平均リスクスコア"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_report_tab(results):
    """レポートタブ"""
    st.subheader("分析レポート")
    
    # レポート生成オプション
    col1, col2 = st.columns(2)
    
    with col1:
        report_format = st.selectbox(
            "レポート形式",
            ["詳細レポート", "サマリーレポート", "エグゼクティブサマリー"]
        )
    
    with col2:
        include_charts = st.checkbox("グラフを含める", value=True)
    
    # レポート生成ボタン
    if st.button("レポート生成"):
        report_content = generate_report(results, report_format, include_charts)
        
        # レポート表示
        st.subheader("生成されたレポート")
        st.markdown(report_content)
        
        # ダウンロードボタン
        st.download_button(
            label="レポートをダウンロード (Markdown)",
            data=report_content,
            file_name=f"risk_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        
        # PDF変換オプション
        if st.button("PDFとしてダウンロード"):
            pdf_data = convert_to_pdf(report_content)
            if pdf_data:
                st.download_button(
                    label="PDFをダウンロード",
                    data=pdf_data,
                    file_name=f"risk_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # 分析結果のJSONエクスポート
    st.subheader("データエクスポート")
    
    if st.button("分析結果をJSONでエクスポート"):
        json_data = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            label="JSONをダウンロード",
            data=json_data,
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def get_risk_level(score):
    """リスクスコアからリスクレベルを判定"""
    if score >= 8:
        return "高リスク"
    elif score >= 6:
        return "中リスク"
    elif score >= 4:
        return "低リスク"
    else:
        return "極低リスク"

def generate_report(results, report_format, include_charts):
    """レポート生成"""
    report = []
    
    # ヘッダー
    report.append("# リスク分析レポート")
    report.append(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report.append("")
    
    # エグゼクティブサマリー
    report.append("## エグゼクティブサマリー")
    overall_risk = results.get('overall_risk_score', 0)
    risk_level = get_risk_level(overall_risk)
    report.append(f"- 総合リスクスコア: {overall_risk:.1f}/10 ({risk_level})")
    report.append(f"- 分析対象企業数: {len(results.get('company_analyses', []))}")
    report.append("")
    
    if report_format == "エグゼクティブサマリー":
        return "\n".join(report)
    
    # 主要な発見事項
    report.append("## 主要な発見事項")
    key_findings = results.get('key_findings', [])
    if key_findings:
        for i, finding in enumerate(key_findings, 1):
            report.append(f"{i}. {finding}")
    else:
        report.append("特筆すべき発見事項はありません。")
    report.append("")
    
    if report_format == "サマリーレポート":
        return "\n".join(report)
    
    # 詳細分析（詳細レポートの場合のみ）
    report.append("## 詳細分析結果")
    company_analyses = results.get('company_analyses', [])
    
    for i, analysis in enumerate(company_analyses):
        company_name = analysis.get('company_name', f'企業{i+1}')
        report.append(f"### {company_name}")
        
        # 基本情報
        company_info = analysis.get('company_info', {})
        if company_info:
            report.append("**基本情報:**")
            for key, value in company_info.items():
                report.append(f"- {key}: {value}")
        
        # リスクスコア
        risk_score = analysis.get('risk_score', 0)
        report.append(f"**リスクスコア:** {risk_score:.1f}/10")
        
        # リスク要因
        risk_factors = analysis.get('risk_factors', [])
        if risk_factors:
            report.append("**特定されたリスク要因:**")
            for factor in risk_factors:
                report.append(f"- {factor.get('type', '不明')}: {factor.get('description', 'N/A')}")
        
        report.append("")
    
    return "\n".join(report)

def convert_to_pdf(markdown_content):
    """MarkdownをPDFに変換"""
    try:
        # 一時的にMarkdownファイルを作成
        temp_md_path = f"/tmp/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        temp_pdf_path = temp_md_path.replace('.md', '.pdf')
        
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # manus-md-to-pdfユーティリティを使用してPDFに変換
        import subprocess
        result = subprocess.run(
            ['manus-md-to-pdf', temp_md_path, temp_pdf_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            with open(temp_pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            # 一時ファイルを削除
            import os
            os.remove(temp_md_path)
            os.remove(temp_pdf_path)
            
            return pdf_data
        else:
            st.error(f"PDF変換エラー: {result.stderr}")
            return None
    
    except Exception as e:
        st.error(f"PDF変換中にエラーが発生しました: {str(e)}")
        return None

# サンプル結果データ生成関数（テスト用）
def generate_sample_results():
    """サンプル分析結果を生成"""
    return {
        'overall_risk_score': 6.5,
        'risk_categories': {
            '財務リスク': 7.2,
            '運営リスク': 5.8,
            '市場リスク': 6.9,
            'コンプライアンスリスク': 4.3,
            '技術リスク': 7.5
        },
        'key_findings': [
            '複数の企業で財務指標の悪化が確認されました',
            '技術リスクが業界平均を上回っています',
            'コンプライアンス体制は比較的良好です'
        ],
        'company_analyses': [
            {
                'company_name': 'サンプル企業A',
                'company_info': {
                    '業界': 'IT',
                    '従業員数': '500-1000',
                    '設立年': '2010'
                },
                'risk_score': 7.2,
                'risk_factors': [
                    {
                        'type': '財務リスク',
                        'severity': '高',
                        'description': '売上高の減少傾向が続いています',
                        'mitigation': '新規事業の開拓と既存事業の効率化'
                    },
                    {
                        'type': '技術リスク',
                        'severity': '中',
                        'description': 'システムの老朽化が進んでいます',
                        'mitigation': 'システムの段階的な更新計画の策定'
                    }
                ],
                'analysis_logs': [
                    '財務データの分析を開始',
                    '売上高の減少傾向を検出',
                    '技術インフラの評価を実施',
                    'リスク評価を完了'
                ]
            },
            {
                'company_name': 'サンプル企業B',
                'company_info': {
                    '業界': '製造業',
                    '従業員数': '1000+',
                    '設立年': '1995'
                },
                'risk_score': 5.8,
                'risk_factors': [
                    {
                        'type': '市場リスク',
                        'severity': '中',
                        'description': '競合他社の台頭により市場シェアが減少',
                        'mitigation': '差別化戦略の強化と新市場の開拓'
                    }
                ],
                'analysis_logs': [
                    '市場データの分析を開始',
                    '競合分析を実施',
                    'リスク評価を完了'
                ]
            }
        ]
    }

if __name__ == "__main__":
    # テスト用のサンプルデータを設定
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = generate_sample_results()
    
    show_results_page()

