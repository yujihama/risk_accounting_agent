import os
import uuid
import json
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, END

import streamlit as st

import PyPDF2

# --- 1. 型定義: 分析のインプットと状態を厳密に定義 ---
class CompanyInput(BaseModel):
    """子会社ごとの入力情報を定義するモデル"""
    company_id: str
    company_name: str
    summary: str  # 例: "主力事業は電子部品の製造・販売"
    key_metrics: Dict[str, str] # 例: {"売上高": "100億円", "営業利益": "10億円"}
    abnormal_indicators: Dict[str, str] # 例: {"在庫回転期間": "前年比50%悪化"}

class RiskCandidate(BaseModel):
    id: str = Field(default_factory=lambda: f"RC_{uuid.uuid4().hex[:4]}")
    description: str
    source_metric: str

class RiskCandidates(BaseModel):
    candidates: List[RiskCandidate]

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="調査の根拠となる要素に分解された、検索に適したキーワードやキーフレーズのリスト。")

class RiskAssessment(BaseModel):
    certainty: str
    reasoning: str
    is_conclusive: bool
    next_question: Optional[str] = None

class FinalAssessment(BaseModel):
    company_id: str
    candidate_id: str
    certainty: str
    reasoning: str
    evidence_docs: List[str]

class GraphState(BaseModel):
    # --- 全体管理（マクロレベル）の状態 ---
    all_companies_input: List[CompanyInput]
    common_qualitative_docs: List[Document] # 全社共通の定性情報
    current_company_index: int = 0
    final_assessments: List[FinalAssessment] = []
    
    # --- 個別会社分析（ミクロレベル）の状態 ---
    current_company_input: Optional[CompanyInput] = None
    current_analysis_summary: str = ""
    retriever_qualitative: Optional[Any] = None
    retriever_quantitative: Optional[Any] = None
    log: List[str] = []
    risk_candidates: List[RiskCandidate] = []
    current_candidate_index_micro: int = 0
    current_investigation_queries: List[str] = []
    gathered_evidence: List[Document] = []
    investigation_depth: int = 0
    max_depth: int = 2
    last_assessment: Optional[RiskAssessment] = None
    company_reports: Dict[str, str] = {}
    # ユーザーが指定したPDF格納フォルダのパス（任意）。
    docs_folder_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# --- 2. プロンプト: 各エージェントの思考を定義 ---
PROMPT_TEMPLATE_INITIAL_SUMMARY = """
あなたはシニア財務アナリストです。
以下は、とある子会社の概要と財務指標です。この情報全体を俯瞰し、最も優先的に調査すべきリスクの概要を日本語で整理してください。
この結果は後続の調査作業のインプットとなります。

## 会社概要
社名: {company_name}
事業概要: {summary}

## 主要財務指標
{key_metrics}

## 検出された異常指標
{abnormal_indicators}
"""

PROMPT_TEMPLATE_CANDIDATE_GENERATOR = """
あなたは優秀なリスクアナリストです。
以下の財務指標の異常値から、想定される事業上のリスク候補を具体的に3つ程度挙げてください。

## 異常指標
{metrics_summary}
"""

PROMPT_TEMPLATE_QUERY_DECOMPOSER = """
あなたは熟練した財務分析の専門家です。
以下のリスクシナリオの文章を分析し、その内容を検証するためにRAG検索で使うためのキーフレーズを3つ程度抽出してください。
各キーフレーズはそれぞれ単独で意味として独立するようにください。
生成されたキーフレーズはベクトル化され類似度が高いチャンクが取得されます。有効に取得できるよう、キーフレーズは省略せずに具体的に生成してください。

## 分析対象の文章
{candidate_description}
"""

PROMPT_TEMPLATE_EVALUATOR_DEEP_DIVE = """
あなたは熟練した財務分析の専門家です。
以下のリスク候補について、集められた証跡情報を基に、現時点での評価を行ってください。
証跡情報には、[定性情報]として議事録やレポートの抜粋と、[定量情報]として具体的な財務指標が含まれています。
これらを必ず突き合わせて、多角的に評価してください。

## 評価対象のリスク候補
{candidate_description}

## これまでに集まった証跡情報
{evidence}

### あなたのタスク
1.  現在の情報だけで、リスクの確度（high・middle・low）を判断してください。
2.  この評価で結論が出たと判断できるか（`is_conclusive`）を判定してください。懐疑的な目線で判断してください。
3.  まだ結論が出ていない場合、次は何を調べればより確実な結論を導けるか、具体的な質問（`next_question`）を考えてください。
4.  上記の回答をするに至った思考過程を整理して次のアクションを説明してください。(`reasoning`)
"""

PROMPT_TEMPLATE_COMPANY_REPORT = """
あなたはCFOに報告を行う経理財務の専門家、監査人です。
以下は子会社について調査したリスク評価の最終結果です。

この調査結果全体を要約し、経営層が意思決定に使えるように、簡潔かつ示唆に富んだ最終報告書を作成してください。

## 分析対象企業
{company_name} ({company_id})

## 調査されたリスク評価一覧
{assessments_summary}

### あなたのタスク
- 調査結果の要点をまとめる。
- 最も注意すべき重大なリスクを特定する。
- 経営層が取るべき次のアクションについて、具体的な提言を行う。
- 全体をプロフェッショナルな報告書の形式で出力する。
"""

# --- 3. ノードの実装 ---

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
embeddings = OpenAIEmbeddings()

# すべてのprintをGraphState.logにも蓄積するヘルパ

def log_and_print(state: "GraphState", message: str):
    """printと同時にGraphState.logへも追記する"""
    print(message)
    # GraphState.log はミュータブルなリストなので直接追加する
    if hasattr(state, "log"):
        state.log.append(message)
    if not message.startswith("---"):
        # --- 画面見切れ防止のため、長い行を折り返すCSSを一度だけ挿入 ---
        if "code_wrap_style_injected" not in st.session_state:
            st.markdown(
                """
                <style>
                /* st.code 内の <pre> 要素に適用 */
                div[data-testid="stCodeBlock"] pre,
                div[data-testid="stCode"] pre {
                    white-space: pre-wrap;  /* 行を折り返す */
                    word-break: break-word; /* 長い単語も折り返す */
                    overflow-x: auto;       /* 必要に応じて横スクロール */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.session_state["code_wrap_style_injected"] = True
        # 実際のログをコードブロックとして表示
        st.code(message)

def load_initial_data_node(state: GraphState) -> Dict[str, Any]:
    """全社共通の定性情報をロードするノード"""
    log_and_print(state, "--- ノード実行: ドキュメントのロード ---")
    
    # # 1. 既存のスタティックデータ
    # common_docs = [
    #     # --- 既存データ ---
    #     Document(page_content="経営会議議事録 2025/05/15: 製品Xの市場からのフィードバックが芳しくないとの報告があった。特に若年層からの支持が得られていない。", metadata={"source": "経営会議議事録", "date": "2025-05-15"}),
    #     Document(page_content="取締役会議事録 2025/06/10: 競合B社の新製品が好調で、製品Xのシェアを奪っている。価格戦略の見直しが急務である。", metadata={"source": "取締役会議事録", "date": "2025-06-10"}),
    #     Document(page_content="品質会議議事録 2025/04/20: 製品Xの初期ロットにおいて、部品Aの軽微な不具合が報告されたが、安全性に問題はないと判断された。", metadata={"source": "品質会議議事録", "date": "2025-04-20"}),
    #     Document(page_content="週次営業報告 2025/06/20: 主要な販売代理店からの製品Xの発注が停止している。理由は競合製品への乗り換えと聞いている。", metadata={"source": "週次営業報告", "date": "2025-06-20"}),

    #     # --- 追加データ (製品・販売) ---
    #     Document(page_content="製品X販売促進レポート 2025/03/31: 春のキャンペーンの売上効果は限定的で、目標の60%に留まった。", metadata={"source": "販売促進レポート", "date": "2025-03-31"}),
    #     Document(page_content="顧客サポートセンター月次レポート 2025/05/31: 製品Xに関する問い合わせのうち、30%が操作性の複雑さに関するものだった。", metadata={"source": "顧客サポートレポート", "date": "2025-05-31"}),
    #     Document(page_content="SNS分析レポート 2025/06/05: インフルエンサーマーケティング施策後、製品Xの認知度は向上したが、購買意欲の向上には繋がっていない。", metadata={"source": "SNS分析レポート", "date": "2025-06-05"}),
    #     Document(page_content="ECサイトA/Bテスト結果 2025/04/10: 製品ページのレイアウト変更によるコンバージョン率の改善は見られなかった。", metadata={"source": "A/Bテスト結果", "date": "2025-04-10"}),
    #     Document(page_content="エリア別売上報告（関西） 2025/05/25: 関西エリアでの製品Xの売上が前月比15%減。競合B社の攻勢が強い。", metadata={"source": "エリア別売上報告", "date": "2025-05-25"}),

    #     # --- 追加データ (競合・市場) ---
    #     Document(page_content="競合分析レポート 2025/02/15: 競合C社が新技術に関する特許を出願。3年以内に市場に影響を与える可能性がある。", metadata={"source": "競合分析レポート", "date": "2025-02-15"}),
    #     Document(page_content="市場調査レポート 2025/01/20: 当社のターゲット市場は、サステナビリティへの関心が高まっているが、製品Xは環境配慮のアピールが弱い。", metadata={"source": "市場調査レポート", "date": "2025-01-20"}),
    #     Document(page_content="外部ニュース 2025/06/15: 新興企業D社が、低価格な類似製品でクラウドファンディングに成功。新たな脅威となる可能性がある。", metadata={"source": "外部ニュース", "date": "2025-06-15"}),

    #     # --- 追加データ (サプライチェーン・製造) ---
    #     Document(page_content="調達部門レポート 2025/03/05: 主要部品Aの供給元であるE国で政情不安が高まっており、供給遅延のリスクがある。", metadata={"source": "調達部門レポート", "date": "2025-03-05"}),
    #     Document(page_content="工場稼働状況報告 2025/04/30: 製造ライン2で頻発する軽微なトラブルにより、生産効率が計画値を5%下回っている。", metadata={"source": "工場稼働状況報告", "date": "2025-04-30"}),
    #     Document(page_content="物流パートナー評価 2025/05/10: 物流委託先F社の配送遅延率が上昇傾向。顧客満足度低下の一因となっている可能性がある。", metadata={"source": "物流パートナー評価", "date": "2025-05-10"}),
    #     Document(page_content="原材料市況レポート 2025/06/01: 製品Xの主要な原材料であるGの価格が、天候不順により3ヶ月で20%高騰している。", metadata={"source": "原材料市況レポート", "date": "2025-06-01"}),

    #     # --- 追加データ (人事・組織) ---
    #     Document(page_content="人事異動通知 2025/02/01: 製品X開発チームのリーダーである田中氏が、一身上の都合により退職。", metadata={"source": "人事異動通知", "date": "2025-02-01"}),
    #     Document(page_content="従業員満足度調査 2025/03/20: 開発部門のワークライフバランスに対する満足度が全社平均を大きく下回っている。", metadata={"source": "従業員満足度調査", "date": "2025-03-20"}),
    #     Document(page_content="社内通達 2025/05/20: 業績不振のため、全社的に経費削減を強化する。特に広告宣伝費は前年比30%削減を目標とする。", metadata={"source": "社内通達", "date": "2025-05-20"}),

    #     # --- 追加データ (財務・法務) ---
    #     Document(page_content="四半期財務予測（修正） 2025/04/05: 第2四半期の売上予測を、製品Xの販売不振を理由に当初計画の85%に下方修正する。", metadata={"source": "四半期財務予測", "date": "2025-04-05"}),
    #     Document(page_content="法務部門からの警告 2025/05/08: 製品Xの広告表現について、景品表示法に抵触する恐れがあるとの指摘あり。直ちに修正が必要。", metadata={"source": "法務部門からの警告", "date": "2025-05-08"}),
    #     Document(page_content="内部監査報告 2025/06/18: 一部の経費申請において、承認プロセスが遵守されていない事例が散見された。", metadata={"source": "内部監査報告", "date": "2025-06-18"}),
    #     Document(page_content="資金調達会議議事録 2025/03/12: 新規プロジェクトのための追加融資について、銀行側は製品Xの将来性を懸念しており、回答を保留。", metadata={"source": "資金調達会議議事録", "date": "2025-03-12"}),

    #     # --- 追加データ (その他) ---
    #     Document(page_content="ITシステム障害報告 2025/04/22: 基幹システムが半日にわたり停止。受注処理に大幅な遅れが発生した。", metadata={"source": "ITシステム障害報告", "date": "2025-04-22"}),
    #     Document(page_content="プレスリリース 2024/12/15: 製品Xが「グッドデザイン賞2024」を受賞しました。", metadata={"source": "プレスリリース", "date": "2024-12-15"}),
    #     Document(page_content="株主総会質疑応答 2025/06/20: 株主から、製品Xの競争力の低下について厳しい質問があった。", metadata={"source": "株主総会質疑応答", "date": "2025-06-20"}),
    #     Document(page_content="コンサルティングレポート要約 2025/01/30: 市場のデジタルシフトが加速しており、当社のオンライン販売チャネルの強化が不可欠である。", metadata={"source": "コンサルティングレポート", "date": "2025-01-30"}),
    #     Document(page_content="特許侵害の可能性に関する通知 2025/06/05: 競合A社より、製品Xが自社特許を侵害している可能性があるとの警告書を受領。", metadata={"source": "外部通知", "date": "2025-06-05"}),
    #     Document(page_content="為替レート変動に関するメモ 2025/05/30: 急激な円安により、海外からの部品調達コストが15%増加している。", metadata={"source": "財務メモ", "date": "2025-05-30"}),
    # ]

    # 2. ユーザーが指定したフォルダ配下の PDF を読み込み
    common_docs = []
    if state.docs_folder_path and os.path.isdir(state.docs_folder_path):
        pdf_files = [
            os.path.join(state.docs_folder_path, f)
            for f in os.listdir(state.docs_folder_path)
            if f.lower().endswith(".pdf")
        ]

        log_and_print(state, f"指定フォルダから {len(pdf_files)} 件のPDFを検出")

        for pdf_path in pdf_files:
            try:
                reader = PyPDF2.PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() or ""
                    except Exception:
                        # ページ単位の抽出失敗は無視して続行
                        continue

                if text.strip():
                    # 200文字でチャンク化 50文字重複
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=200,
                        chunk_overlap=50,
                        length_function=len,
                        separators=["\n\n", "\n", "。", "、", " ", ""]
                    )
                    
                    chunks = text_splitter.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(pdf_path) + "_" + str(i), 
                                "path": pdf_path,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        )
                        common_docs.append(doc)
                    # log_and_print(state, f"✅ 取り込み成功: {os.path.basename(pdf_path)}")
                else:
                    log_and_print(state, f"⚠️ テキスト抽出なし: {os.path.basename(pdf_path)}")
            except Exception as e:
                log_and_print(state, f"❌ PDF読み取りエラー ({os.path.basename(pdf_path)}): {str(e)}")

    elif state.docs_folder_path:
        # パスは指定されたが存在しない
        log_and_print(state, f"❌ 指定フォルダが見つかりません: {state.docs_folder_path}")

    # 3. 結果を返却
    return {"common_qualitative_docs": common_docs, "log": state.log}


def setup_company_analysis_node(state: GraphState) -> Dict[str, Any]:
    """会社ごとの分析を開始し、ハイブリッドRAGストアを構築するノード"""
    company_index = state.current_company_index
    company_input = state.all_companies_input[company_index]
    log_and_print(state, f"\n--- ノード実行: [{company_input.company_name}] の分析準備開始 ---")

    # 定量情報をベクトル化する
    quantitative_docs = []
    for k, v in company_input.key_metrics.items():
        quantitative_docs.append(Document(page_content=f"[定量情報] 主要指標 {k}: {v}", metadata={"source": "財務データ", "company": company_input.company_name}))
    for k, v in company_input.abnormal_indicators.items():
        quantitative_docs.append(Document(page_content=f"[定量情報] 異常指標 {k}: {v}", metadata={"source": "財務データ", "company": company_input.company_name}))

    # 定性情報と結合してハイブリッドストアを構築
    hybrid_docs = state.common_qualitative_docs + quantitative_docs
    vector_store_qualitative = FAISS.from_documents(state.common_qualitative_docs, embeddings)
    vector_store_quantitative = FAISS.from_documents(quantitative_docs, embeddings)

    retriever_qualitative = vector_store_qualitative.as_retriever(search_kwargs={"k": 2})
    retriever_quantitative = vector_store_quantitative.as_retriever(search_kwargs={"k": 5})

    log_message = f"[{company_input.company_name}] のハイブリッドRAGストアを構築しました ({len(hybrid_docs)}件)。"
    log_and_print(state, log_message)

    # 分析サマリーを生成
    summary_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_INITIAL_SUMMARY)
    summary_chain = summary_prompt | llm
    analysis_summary = summary_chain.invoke({
        "company_name": company_input.company_name,
        "summary": company_input.summary,
        "key_metrics": json.dumps(company_input.key_metrics, ensure_ascii=False),
        "abnormal_indicators": json.dumps(company_input.abnormal_indicators, ensure_ascii=False)
    }).content
    
    log_and_print(state, f"[thought] {analysis_summary}")

    return {
        "current_company_input": company_input,
        "retriever_qualitative": retriever_qualitative,
        "retriever_quantitative": retriever_quantitative,
        "log": state.log,
        "current_analysis_summary": analysis_summary
    }

def generate_candidates_node(state: GraphState) -> Dict[str, Any]:
    """分析サマリーからリスク候補を生成"""
    log_and_print(state, "--- ノード実行: リスク候補の生成 ---")
    structured_llm_cand = llm.with_structured_output(RiskCandidates)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_CANDIDATE_GENERATOR)
    chain = prompt | structured_llm_cand
    # Change 'summary' to 'metrics_summary' to match the prompt template
    response = chain.invoke({"metrics_summary": state.current_analysis_summary})
    log_and_print(state, "リスク分析の観点を生成しました。")
    for cand in response.candidates:
        cand.source_metric = state.current_analysis_summary
        log_and_print(state, f"観点: {cand.description}")
    
    return {
        "risk_candidates": response.candidates,
        "log": state.log,
        "current_candidate_index_micro": 0 # ミクロループのインデックスを初期化
    }

def start_investigation_node(state: GraphState) -> Dict[str, Any]:
    candidate_index = state.current_candidate_index_micro
    candidate = state.risk_candidates[candidate_index]
    log_and_print(state, f"--- ノード実行: [{state.current_company_input.company_name}] のリスク調査開始 ---")
    log_and_print(state, f"調査対象: ({candidate.id}) {candidate.description}")
    # log_and_print(state, f"観点({candidate.id})の調査を開始。")
    return {
        "gathered_evidence": [],
        "investigation_depth": 0,
        "log": state.log
    }

def decompose_query_node(state: GraphState) -> Dict[str, Any]:
    log_and_print(state, "--- ノード実行: 検索クエリの分解 ---")
    if state.investigation_depth == 0:
        source_text = state.risk_candidates[state.current_candidate_index_micro].description
    else:
        source_text = state.last_assessment.next_question or ""

    structured_llm_decomp = llm.with_structured_output(SearchQueries)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_QUERY_DECOMPOSER)
    chain = prompt | structured_llm_decomp
    # Change 'source_text' to 'candidate_description' to match the prompt template
    response = chain.invoke({"candidate_description": source_text})
    log_and_print(state, f"[thought] 分解された観点: {response.queries}")
    return {"current_investigation_queries": response.queries, "log": state.log}

def rag_search_node(state: GraphState) -> Dict[str, Any]:
    log_and_print(state, f"--- ノード実行: ハイブリッドRAG検索 (深さ:{state.investigation_depth + 1}) ---")
    # retriever = state.retriever # Removed this line
    retriever_qualitative = state.retriever_qualitative # Use qualitative retriever
    retriever_quantitative = state.retriever_quantitative # Use quantitative retriever

    queries = state.current_investigation_queries

    all_retrieved_docs = []
    for query in queries:
        log_and_print(state, f"検索観点: {query}")
        # Query both retrievers
        retrieved_docs_qualitative = retriever_qualitative.invoke(query)
        retrieved_docs_quantitative = retriever_quantitative.invoke(query)
        all_retrieved_docs.extend(retrieved_docs_qualitative)
        all_retrieved_docs.extend(retrieved_docs_quantitative)

        # タイトルのみを表示
        log_and_print(state, f"[thought] 関連情報: [{"],[".join([doc.metadata['source'] for doc in retrieved_docs_qualitative])}]")

    summary_msg = f"{len(queries)}個の観点で合計{len(all_retrieved_docs)}件の情報を発見"
    log_and_print(state, summary_msg)

    # 重複を排除して証拠を蓄積
    existing_contents = {doc.page_content for doc in state.gathered_evidence}
    new_evidence = state.gathered_evidence
    for doc in all_retrieved_docs:
        if doc.page_content not in existing_contents:
            new_evidence.append(doc)

    return {"gathered_evidence": new_evidence, "log": state.log}

def evaluate_and_decide_node(state: GraphState) -> Dict[str, Any]:
    log_and_print(state, "--- ノード実行: 評価 ---")
    candidate = state.risk_candidates[state.current_candidate_index_micro]
    evidence_text = "\n\n".join([doc.page_content for doc in state.gathered_evidence])

    structured_llm_eval = llm.with_structured_output(RiskAssessment)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EVALUATOR_DEEP_DIVE)
    chain = prompt | structured_llm_eval
    
    assessment = chain.invoke({
        "candidate_description": candidate.description,
        "evidence": evidence_text
    })
    
    log_and_print(state, f"[thought] {assessment.reasoning}")
    
    return {
        "last_assessment": assessment,
        "investigation_depth": state.investigation_depth + 1,
        "log": state.log
    }

def finalize_assessment_node(state: GraphState) -> Dict[str, Any]:
    log_and_print(state, "--- ノード実行: 評価の最終化 ---")
    candidate = state.risk_candidates[state.current_candidate_index_micro]
    last_assessment = state.last_assessment
    
    final_assessment = FinalAssessment(
        company_id=state.current_company_input.company_id,
        candidate_id=candidate.id,
        certainty=last_assessment.certainty,
        reasoning=last_assessment.reasoning,
        evidence_docs=list(set([doc.page_content for doc in state.gathered_evidence]))
    )
    
    return {
        "final_assessments": state.final_assessments + [final_assessment],
        "current_candidate_index_micro": state.current_candidate_index_micro + 1,
        "log": state.log
    }

def generate_company_report_node(state: GraphState) -> Dict[str, Any]:
    company_input = state.current_company_input
    log_and_print(state, f"--- ノード実行: [{company_input.company_name}] の最終レポート生成 ---")

    # この会社に関する評価のみを抽出
    company_assessments = [
        assess for assess in state.final_assessments 
        if assess.company_id == company_input.company_id
    ]

    # LLMに渡すために評価結果を整形
    assessments_summary_text = ""
    for i, assessment in enumerate(company_assessments, 1):
        assessments_summary_text += f"\n--- リスク評価 {i} ---\n"
        assessments_summary_text += f"ID: {assessment.candidate_id}\n"
        assessments_summary_text += f"確度: {assessment.certainty}\n"
        assessments_summary_text += f"判断理由: {assessment.reasoning}\n"
        assessments_summary_text += f"根拠資料: {assessment.evidence_docs}\n"

    # レポート生成の実行
    report_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_COMPANY_REPORT)
    report_chain = report_prompt | llm
    
    report_content = report_chain.invoke({
        "company_name": company_input.company_name,
        "company_id": company_input.company_id,
        "assessments_summary": assessments_summary_text
    }).content

    # 生成されたレポートをコンソールに表示し、状態に保存
    # header_line = "\n" + "="*20 + f" [{company_input.company_name}] 最終レポート " + "="*20
    # footer_line = "="*60 + "\n"
    # log_and_print(state, header_line)
    # log_and_print(state, report_content)
    # log_and_print(state, footer_line)

    updated_reports = state.company_reports.copy()
    updated_reports[company_input.company_id] = report_content
    
    return {"company_reports": updated_reports, "log": state.log}


def prepare_for_next_company_node(state: GraphState) -> Dict[str, Any]:
    """次の会社の分析準備のため、会社のインデックスをインクリメントする"""
    log_and_print(state, "--- ノード実行: 次の会社の準備 ---")
    return {"current_company_index": state.current_company_index + 1, "log": state.log}


# --- 4. 条件分岐ロジック ---

def should_deepen_investigation(state: GraphState) -> str:
    """深掘り調査を継続するか、完了するかを判定する（内側ループの判定）"""
    log_and_print(state, "--- ノード実行: 深掘り調査の継続判定 ---")
    if state.last_assessment and state.last_assessment.is_conclusive:
        log_and_print(state, "[thought] 結論が出たため調査を完了")
        return "complete"
    if state.investigation_depth >= state.max_depth:
        log_and_print(state, "[thought] 最大深度に到達。調査を完了")
        return "complete"
    return "continue"

def should_investigate_next_candidate(state: GraphState) -> str:
    """次のリスク候補の調査に移るか、その会社の分析を完了するかを判定する"""
    log_and_print(state, "--- ノード実行: 次の[リスク候補]の調査判定 ---")
    if state.current_candidate_index_micro < len(state.risk_candidates):
        # log_and_print(state, f"判定: [{state.current_company_input.company_name}] の次のリスク候補の調査へ。")
        return "continue"
    # log_and_print(state, f"判定: [{state.current_company_input.company_name}] の全リスク候補の調査が完了。")
    return "complete"

def should_analyze_next_company(state: GraphState) -> str:
    """次の会社の分析に移るか、全体を終了するかを判定する"""
    log_and_print(state, "--- ノード実行: 次の[会社]の分析判定 ---")
    if state.current_company_index < len(state.all_companies_input):
        log_and_print(state, f"[thought] 次の会社 ({state.current_company_index + 1}社目) の分析へ")
        return "continue"
    log_and_print(state, "[thought] 全ての会社の分析が完了")
    return "end"

# --- 5. グラフの再構築 ---

graph_builder = StateGraph(GraphState)

# ノードの追加
graph_builder.add_node("load_initial_data", load_initial_data_node)
graph_builder.add_node("setup_company_analysis", setup_company_analysis_node)
graph_builder.add_node("generate_candidates", generate_candidates_node)
graph_builder.add_node("start_investigation", start_investigation_node)
graph_builder.add_node("decompose_query", decompose_query_node)
graph_builder.add_node("rag_search", rag_search_node)
graph_builder.add_node("evaluate_and_decide", evaluate_and_decide_node)
graph_builder.add_node("finalize_assessment", finalize_assessment_node)
graph_builder.add_node("generate_company_report", generate_company_report_node) 
graph_builder.add_node("prepare_for_next_company", prepare_for_next_company_node)

# ミクロループの開始を制御するルーターノード
graph_builder.add_node("micro_loop_router", lambda state: state)

# グラフのエッジ定義
graph_builder.set_entry_point("load_initial_data")

# 最初の会社分析のセットアップ
graph_builder.add_edge("load_initial_data", "setup_company_analysis")

# 会社ごとの分析フロー
graph_builder.add_edge("setup_company_analysis", "generate_candidates")
graph_builder.add_edge("generate_candidates", "micro_loop_router")

# ミクロループ（リスク候補ごとの調査）
graph_builder.add_conditional_edges(
    "micro_loop_router",
    should_investigate_next_candidate,
    {
        "continue": "start_investigation",
        "complete": "generate_company_report"
    }
)
graph_builder.add_edge("start_investigation", "decompose_query")

# 深掘りループ
graph_builder.add_edge("decompose_query", "rag_search")
graph_builder.add_edge("rag_search", "evaluate_and_decide")
graph_builder.add_conditional_edges(
    "evaluate_and_decide",
    should_deepen_investigation,
    {
        "continue": "decompose_query",
        "complete": "finalize_assessment"
    }
)
# 評価完了後は、次のリスクを判断するルーターに戻る
graph_builder.add_edge("finalize_assessment", "micro_loop_router")

# レポート生成が終わったら、次の会社の準備に移る
graph_builder.add_edge("generate_company_report", "prepare_for_next_company")

# マクロループ（次の会社へ）
graph_builder.add_conditional_edges(
    "prepare_for_next_company",
    should_analyze_next_company,
    {
        "continue": "setup_company_analysis", # 次の会社のセットアップへ
        "end": END # 全ての会社の分析が完了
    }
)

# グラフをコンパイル
app = graph_builder.compile()

# --- 6. 実行 ---
if __name__ == "__main__":
    # サンプル入力データ
    initial_input_data = {
        "all_companies_input": [
            CompanyInput(
                company_id="A001",
                company_name="子会社A",
                summary="主力事業は電子部品（製品X）の製造・販売。特にコンシューマー向け製品に強み。",
                key_metrics={"売上高": "100億円", "営業利益": "5億円"},
                abnormal_indicators={"在庫回転期間": "前年比50%悪化", "返品率": "3倍に増加"}
            ),
            CompanyInput(
                company_id="B002",
                company_name="子会社B",
                summary="産業用ソフトウェア（製品Y）の開発・販売。安定した顧客基盤を持つが、近年成長が鈍化。",
                key_metrics={"売上高": "50億円", "営業利益": "10億円"},
                abnormal_indicators={"新規契約数": "前年比20%減", "解約率": "5%上昇"}
            )
        ],
        # common_qualitative_docs フィールドを初期状態で追加する
        "common_qualitative_docs": []
    }
    
    final_state = app.invoke(initial_input_data,config={"recursion_limit": 100})

