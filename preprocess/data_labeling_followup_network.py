# -*- coding: utf-8 -*-
"""Demo script for LLM-assisted follow-up network labeling.

In the full experiment pipeline, the raw annotations were produced by three
LLM-based labeling runs. This script is kept as a compact demo for constructing
local candidate edges (Q1, A1) -> Q2 and assigning follow-up labels. If you
reuse it, please manually adjust the model name, prompt, and API configuration
to match your local setup.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI
from pandas.errors import EmptyDataError


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_INPUT_FILE = "full_dataset_CNRDS_30firms.csv"
DEFAULT_OUTPUT_FILE = "./train data/labeled_data_Deepseek.csv"
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MAX_CANDIDATES = 0
DEFAULT_RANDOM_STATE = 42
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_SAVE_EVERY = 25
DEFAULT_MAX_API_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 5.0
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5
DEFAULT_EASY_SHIFT_THRESHOLD = 0.80
DEFAULT_BOUNDARY_SHIFT_THRESHOLD = 0.45

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


FOLLOWUP_CUES = {
    "为什么", "为何", "原因", "具体", "进一步", "继续", "上述", "前述", "这个", "这些",
    "该", "是否", "如何", "能否", "请问", "说明", "解释", "补充", "详细", "多久",
    "什么时候", "何时", "多少", "吗", "呢",
}

ANSWER_COMPLAINT_CUES = {
    "没回话", "没回复", "没回答", "不回答", "未回答", "没有回答", "不送出",
    "回答我的问题", "回答问题", "选择性", "效率太低", "还没开始",
}

VAGUE_ANSWER_CUES = {
    "积极", "努力", "持续", "稳步", "进一步", "未来", "后续", "视情况", "根据情况",
    "暂不", "暂无", "不便", "请关注", "以公告为准", "相关规定", "正常推进",
}

STRONG_SHIFT_PATTERNS = (
    "换个话题", "换个问题", "换一个问题", "换个角度", "换个说法", "换一个", "换下一个",
    "换一件事", "换一个话题", "换个别的问题", "换个别的话题", "另一个问题", "另外一个问题",
    "另外一个", "另一个事", "另一个话题", "另外想问", "另外请问", "另外问一下", "另问",
    "另请问", "另请教", "另起一个", "顺便问一下", "顺便再问", "顺手问一下", "顺便问个别的",
    "顺便再请教", "再问一个", "再问一下", "再问点别的", "再请问", "再请教一个", "还有一个问题",
    "还有个问题", "还有个事", "还有一件事", "还有别的问题", "我再问一个", "我还想问", "我另问",
    "我再请问", "无关", "跑题", "偏题", "答非所问", "不相关", "与此无关", "与这个无关",
    "跟这个无关", "这不是我想问的", "新问题", "另一个点", "别的问题", "其他问题",
)

SOFT_SHIFT_PATTERNS = (
    "另外", "此外", "顺便", "另请问", "请问另外", "再请问", "请教一下", "我还想问",
    "我再问一下", "还有个想问", "另外我想问", "再补充问", "再追问一下",
)

QUESTION_PATTERNS = (
    "为什么", "为何", "如何", "怎样", "是否", "有没有", "可不可以", "能否", "什么", "多少",
)


FOLLOWUP_NETWORK_SYSTEM_PROMPT = """
【角色设定】
你是一位金融语用学专家，熟悉上市公司投资者互动平台、业绩说明会问答和投资者追问行为。

【任务目标】
请判断候选边 (Q1, A1) -> Q2 是否构成“追问依赖边”。

这里的“追问依赖”不是单纯同主题，而是 Q2 的提出依赖上一轮 Q1-A1 交互。Yes 包括两类：

一、内容依赖 Content-Dependent：
Q2 针对 A1 中已经出现的事实、数据、解释、承诺、时间安排、模糊表述、回避性说法、逻辑漏洞或矛盾点继续追问、确认、要求解释或施压。

二、交互依赖 Interaction-Dependent：
Q2 针对 A1 的回答状态或回答方式继续互动，例如认为 A1 没回答、回答不充分、回避、敷衍、选择性回答、没有正面回应，于是催促、质疑、追责或要求继续回答。
注意：这类 Q2 即使不复用 A1 的具体词汇，也可以判为 Yes。

【No 的边界】
以下情况判 No：
1. Q2 开启新的独立话题，即使时间上相邻。
2. Q2 只是与 Q1/A1 同属一个大主题，但没有针对 A1 或回答状态发生互动。
3. Q2 不读 A1 也能自然提出，且不是针对 A1 的回避、未答或不充分。
4. Q1 本身包含多个子问题，Q2 只是回到 Q1 的另一个子问题，而不是针对 A1 或回答状态继续互动。
5. 如果 A1 只是笼统、客套、回避或“请关注公告”，但 Q2 提出的是新的独立财务指标、项目、产品、市场、业务、股价或治理问题，且没有明确说“没回答/请继续回答/上述回答/这个问题/该事项”，则判 No。
6. 不要因为 A1 很短、很模糊或很敷衍，就把所有后续问题都判为 Yes；必须能指出 Q2 依赖 A1 的具体内容，或明确针对 A1 的回答缺失/回答方式。
7. 不要仅因为 Q2 仍然问同一位高管、同一家公司、同一行业，或仍处于“业绩/风险/投资/战略”等大主题，就判 Yes。必须证明 Q2 的核心对象、指标、风险点、承诺、解释或未答状态来自 Q1/A1。
8. 如果 Q2 更换了核心对象、年份、报表项目、业务线、产品、子公司、资本市场事件或提问对象，即使 A1 很笼统，也优先判 No，除非 Q2 明确承接 A1 的未答状态或具体内容。

【必须判 Yes 的补充规则】
1. 如果 Q2 重复 Q1、改写 Q1、或继续追问 Q1 中同一个核心问题，而 A1 是“请关注公告/年报/季报/不方便提供/暂不披露/未正面回答/笼统表态”，则判 Yes，dependency_type=Interaction-Dependent，relation_type 优先选 Complaint 或 Challenge。
2. 如果 Q2 的核心对象、公司事项、业务线、风险点、财务指标或经营现象直接来自 Q1 或 A1，并进一步追问原因、数量、措施、影响、时间表、是否正常、是否改善，则判 Yes，dependency_type=Content-Dependent。即使 Q2 没有使用“上述/这个/继续”等显式指代词，也不要误判为新独立问题。
3. 如果 Q1/A1 与 Q2 处在同一条具体经营风险链条或财务政策链条中，例如自然灾害/风险控制/战略转型、扇贝减值/扭亏/生长监测、送转/分红/金融政策、现金流/负债/融资/抵押估值，则优先判为 Yes。注意：“同一链条”必须共享具体对象或具体指标，不能只是同属业绩、风险、战略、投资这类大主题。只有当 Q2 转向完全不同的对象、业务或指标时才判 No。
4. 如果 A1 已经给出一个方向性回答，但没有展开某个关键对象、数量、比例、进度、影响、承诺执行方式或风险后果，而 Q2 正是把这个缺口具体化，那么仍然判 Yes，通常属于 Content-Dependent。常见形式包括追问某个对象的成本比例、产量、持股细节、债务完成情况、重组进展、业务盈利性、外部变量影响等。
5. 如果 Q2 是围绕 A1 中已经出现的具体对象做“继续细化、补充、量化、确认、落实承诺”，而不是开启新对象，那么优先判 Content-Dependent。不要因为 Q2 语气更换成“请问是否”“是不是”“能否承诺”就误判为新话题。

【判断顺序】
1. 先检查是否属于“重复/改写同一问题 + A1 回避或未正面回答”；若是，dependent=Yes, dependency_type=Interaction-Dependent。
2. 再检查 Q2 的核心对象是否来自 Q1 或 A1，并围绕该对象进一步追问原因、数量、措施、影响、时间表、是否正常、是否改善；若是，dependent=Yes, dependency_type=Content-Dependent。
3. 再检查 Q2 是否处于同一条具体经营风险链条或财务政策链条，且共享具体对象或具体指标；若是，dependent=Yes, dependency_type=Content-Dependent。
4. 再检查是否只是同一高管、同一公司、同一行业、同一大主题下的连续提问；若没有共享具体对象/指标/风险点/承诺/解释/未答状态，则判 No。
5. 再判断 Q2 是否是新的独立问题：如果 Q2 的核心对象、指标、项目或业务事项既没有来自 Q1/A1，也没有投诉 A1 未回答，则判 No。
6. 如果 Q2 明确针对 A1 的回答缺失、回避、不充分或回答态度继续追问；若是，dependent=Yes, dependency_type=Interaction-Dependent。
7. 如果以上都不是，dependent=No, dependency_type=No-Dependency。

【relation_type】
若 dependent=Yes，请选择：
- Elaboration：索要更多细节、原因、数据、时间表、计划或解释。常见于对 A1 已提到的具体对象继续细化，例如追问成本比例、产量、持股增减、债务是否完成、重组进程、业务盈利性、外部因素影响，或要求把笼统回答落实成可执行细节。
- Challenge：质疑、反驳、施压、追责，或指出回答不充分/不回答。常见于引用相反数据、质疑 A1 结论依据、指出表述自相矛盾，或者认为 A1 的说法过于乐观、回避现实风险、需要重新说明。
- Clarification：只是在确认自己是否正确理解 A1 的意思，且没有新增对象、没有新增要求、没有追问任务。若 Q2 还要求补充信息、解释原因、量化指标、确认进度或落实承诺，则不要用 Clarification，而应改为 Elaboration 或 Challenge。
- Complaint：主要是在抱怨、催促或质疑“没有回答/没有正面回答”。常见于“怎么没回话”“请具体一些”“到底有没有”“别只说公告”“请继续回答”这类催问，重点是催促 A1 补答或重答。
若 dependent=No，则 relation_type=Topic Shift。

【输出格式】
必须严格输出合法 JSON，不要输出 Markdown：
{
  "dependent": "Yes / No",
  "dependency_type": "Content-Dependent / Interaction-Dependent / No-Dependency",
  "relation_type": "Elaboration / Challenge / Clarification / Complaint / Topic Shift",
  "reason": "50字内说明判断依据"
}

【示例1：内容依赖】
Q1: 贵公司海外市场拓展情况如何？
A1: 欧洲市场营收增长20%，主要得益于德国新设直营渠道。
Q2: 既然德国直营渠道效果显著，今年是否会复制到法国？
输出:
{"dependent":"Yes","dependency_type":"Content-Dependent","relation_type":"Elaboration","reason":"Q2依赖A1中德国直营渠道信息继续追问。"}

【示例2：交互依赖】
Q1: 该项目目前进展如何？
A1: 公司会持续关注并及时披露。
Q2: 怎么没回话呢？
输出:
{"dependent":"Yes","dependency_type":"Interaction-Dependent","relation_type":"Complaint","reason":"Q2针对上一轮回答不充分进行催促。"}

【示例3：内容依赖，针对笼统承诺要求具体化】
Q1: 公司未来有什么规划？
A1: 公司将继续努力提升经营质量。
Q2: 请问公司有何实际行动来回报股民？
输出:
{"dependent":"Yes","dependency_type":"Content-Dependent","relation_type":"Elaboration","reason":"Q2针对A1笼统表述要求具体行动。"}

【示例4：笼统回答后的新独立问题，仍判 No】
Q1: 2012年业绩预计如何？
A1: 公司将努力实现可持续、有质量增长。
Q2: 公司现金流差，连续5年经营净现金流都低于净利润，请问原因是什么？
输出:
{"dependent":"No","dependency_type":"No-Dependency","relation_type":"Topic Shift","reason":"Q2提出新的现金流问题，不依赖A1具体内容。"}

【示例5：回避回答后的明确投诉，判 Yes】
Q1: 公司战略定位是什么？
A1: 请关注公司年报，谢谢。
Q2: 回答问题请具体一些。
输出:
{"dependent":"Yes","dependency_type":"Interaction-Dependent","relation_type":"Complaint","reason":"Q2明确针对A1回答不具体进行催促。"}

【示例6：同一财务政策链条，判 Yes】
Q1: 公司为什么没有送股？
A1: 公司会结合发展阶段和整体金融政策综合考虑。
Q2: 公司刚上市，为什么又进行高比例分红？
输出:
{"dependent":"Yes","dependency_type":"Content-Dependent","relation_type":"Challenge","reason":"Q2围绕A1中的上市阶段和金融政策继续质疑分红安排。"}

【示例7：新话题】
Q1: 公司去年海外市场如何？
A1: 欧洲市场增长20%。
Q2: 请问今年研发投入资本化比例是多少？
输出:
{"dependent":"No","dependency_type":"No-Dependency","relation_type":"Topic Shift","reason":"Q2转向研发投入，不依赖上一轮回答。"}

【示例7B：同一高管或同一公司不等于追问】
Q1: 公司今年有什么发展机遇？
A1: 国家拉动内需和消费升级将带来行业机会。
Q2: 请问王总，公司去年业绩为什么没有达到30%增长？
输出:
{"dependent":"No","dependency_type":"No-Dependency","relation_type":"Topic Shift","reason":"Q2更换为业绩增长新问题，不能仅因同一公司或高管连续提问判Yes。"}

【示例8：重复未答问题，判 Yes】
Q1: 公司海参业务今年是否会有更大增长，具体目标是多少？
A1: 请关注公司后续公告。
Q2: 海参业务今年到底有没有增长目标？
输出:
{"dependent":"Yes","dependency_type":"Interaction-Dependent","relation_type":"Complaint","reason":"Q2重复追问A1未正面回答的同一问题。"}

【示例9：围绕 A1 对象继续追问，判 Yes】
Q1: 自然灾害后公司如何保持经营稳定？
A1: 公司将加强海洋牧场风险控制，降低灾害对扇贝业务的影响。
Q2: 公司是否会推进战略转型来减少养殖风险？
输出:
{"dependent":"Yes","dependency_type":"Content-Dependent","relation_type":"Elaboration","reason":"Q2围绕A1中的灾害风险控制继续追问措施。"}

【示例10：同一经营对象延展追问，判 Yes】
Q1: 公司什么时候可以实现扭亏为盈？
A1: 底播虾夷扇贝业务受存货减值影响效益下降，公司力争全年扭亏。
Q2: 今年扇贝成长正常吗，有没有发现异常情况？
输出:
{"dependent":"Yes","dependency_type":"Content-Dependent","relation_type":"Elaboration","reason":"Q2围绕A1中的扇贝减值和扭亏继续追问经营状态。"}

【示例11：同属大主题但对象已换，判 No】
Q1: 公司如何降低财务费用？
A1: 公司将偿还部分银行贷款，降低利息支出。
Q2: 公司是否有增发融资计划？
输出:
{"dependent":"No","dependency_type":"No-Dependency","relation_type":"Topic Shift","reason":"Q2转向增发融资计划，虽同属财务主题但核心对象已更换。"}
"""


def create_client(args: argparse.Namespace) -> OpenAI:
    if not API_KEY:
        raise ValueError("请先通过环境变量 DEEPSEEK_API_KEY 配置 API Key。")
    return OpenAI(
        api_key=API_KEY,
        base_url=args.base_url,
        max_retries=0,
        timeout=45.0,
    )


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def has_any(text: str, cues: set) -> bool:
    return any(cue in text for cue in cues)


def char_jaccard(left: str, right: str) -> float:
    a = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", clean_text(left)))
    b = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", clean_text(right)))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_terms(text: Any) -> set[str]:
    value = clean_text(text)
    if not value:
        return set()
    value = re.sub(r"[，。；;：:\n\r\t]+", " ", value)
    terms: set[str] = set()
    for chunk in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,10}", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk in {
            "请问", "您好", "谢谢", "你好", "麻烦", "一下", "这个", "那个",
            "是否", "如何", "公司", "董事长", "董秘", "请教", "问题",
        }:
            continue
        terms.add(chunk)
    return terms


def score_topic_shift(q1: str, a1: str, q2: str) -> tuple[float, List[str]]:
    q1_text = clean_text(q1)
    a1_text = clean_text(a1)
    q2_text = clean_text(q2)
    anchor_text = f"{q1_text} {a1_text}"

    score = 0.0
    flags: List[str] = []

    if any(pat in q2_text for pat in STRONG_SHIFT_PATTERNS):
        score += 0.85
        flags.append("strong_pattern")
    elif any(pat in q2_text for pat in SOFT_SHIFT_PATTERNS):
        score += 0.35
        flags.append("soft_pattern")

    anchor_terms = extract_terms(anchor_text)
    q2_terms = extract_terms(q2_text)
    if anchor_terms and q2_terms:
        overlap = len(anchor_terms & q2_terms) / max(1, min(len(anchor_terms), len(q2_terms)))
    else:
        overlap = 0.0

    short_question = len(q2_text) <= 20
    if overlap < 0.05 and short_question:
        score += 0.35
        flags.append("low_overlap_short_q2")
    if overlap == 0.0:
        score += 0.22
        flags.append("no_overlap")
    elif overlap < 0.08:
        score += 0.12
        flags.append("tiny_overlap")

    if short_question and overlap < 0.12 and not any(pat in q2_text for pat in ("继续", "上述", "前面", "刚才", "刚刚", "同上", "关于这个", "这个问题")):
        if any(pat in q2_text for pat in ("另外", "另", "再问", "顺便", "还有", "新问题", "换个", "换一个")):
            score += 0.18
            flags.append("new_topic_short_q2")

    if any(pat in q2_text for pat in ("继续", "上述", "前面", "刚才", "刚刚", "同上", "关于这个", "这个问题")):
        score -= 0.15
        flags.append("continuation_marker")

    if any(pat in q2_text for pat in QUESTION_PATTERNS):
        score += 0.03
        flags.append("question_form")

    if len(q2_text) <= 10:
        score += 0.05
        flags.append("short_q2")
    elif len(q2_text) <= 16:
        score += 0.04
        flags.append("short_q2")
    elif short_question:
        score += 0.03
        flags.append("short_q2")

    if q2_text and q2_text not in q1_text and q2_text not in a1_text:
        score += 0.03

    return float(max(0.0, min(1.0, score))), flags


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="gbk", dtype={"Scode": str, "Year": str})
    except Exception:
        df = pd.read_csv(file_path, encoding="utf-8", dtype={"Scode": str, "Year": str})

    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").copy()
    if "Scode" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"{file_path} 必须包含 Scode 和 Year 列。")

    df["Scode"] = df["Scode"].astype(str).str.strip().str.zfill(6)
    df["Year"] = df["Year"].astype(str).str.strip()
    df["session_id"] = df["Scode"] + "_" + df["Year"]

    sort_cols = ["Scode", "Year"]
    if "Qnumbr" in df.columns:
        sort_cols.append("Qnumbr")
    elif "Date" in df.columns:
        sort_cols.append("Date")
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_candidates(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for session_id, group in df.groupby("session_id", sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            q1 = clean_text(group.loc[i, "Qcntet"])
            a1 = clean_text(group.loc[i, "Acntet"])
            if not q1 or not a1:
                continue
            upper = min(i + 1 + window_size, len(group))
            for j in range(i + 1, upper):
                q2 = clean_text(group.loc[j, "Qcntet"])
                if not q2:
                    continue
                rows.append(
                    {
                        "edge_id": f"{session_id}_{i}_{j}",
                        "session_id": session_id,
                        "Scode": group.loc[i, "Scode"] if "Scode" in group.columns else "",
                        "Year": group.loc[i, "Year"] if "Year" in group.columns else "",
                        "Q1_row": i,
                        "Q2_row": j,
                        "distance": j - i,
                        "Q1_Qnumbr": group.loc[i, "Qnumbr"] if "Qnumbr" in group.columns else "",
                        "Q2_Qnumbr": group.loc[j, "Qnumbr"] if "Qnumbr" in group.columns else "",
                        "Q1": q1,
                        "A1": a1,
                        "Q2": q2,
                    }
                )
    return pd.DataFrame(rows)


def heuristic_score(row: pd.Series) -> float:
    q1 = clean_text(row["Q1"])
    a1 = clean_text(row["A1"])
    q2 = clean_text(row["Q2"])
    distance = float(row.get("distance", 1) or 1)
    follow = float(has_any(q2, FOLLOWUP_CUES))
    complaint = float(has_any(q2, ANSWER_COMPLAINT_CUES))
    vague_answer = float(has_any(a1, VAGUE_ANSWER_CUES))
    overlap = max(char_jaccard(q1, q2), char_jaccard(a1, q2))
    close = 1.0 / (1.0 + max(distance, 1.0))
    short_complaint = float(len(q2) <= 12 and (complaint or "为什么" in q2 or "怎么" in q2))
    return min(1.0, 0.26 * follow + 0.30 * complaint + 0.16 * vague_answer + 0.16 * overlap + 0.08 * close + 0.20 * short_complaint)


def sample_candidates(df: pd.DataFrame, max_candidates: int, random_state: int) -> pd.DataFrame:
    if max_candidates <= 0 or len(df) <= max_candidates:
        return df.reset_index(drop=True)

    scored = df.copy()
    scored["prompt_test_score"] = scored.apply(heuristic_score, axis=1)
    high_n = max_candidates // 2
    random_n = max_candidates - high_n
    high = scored.nlargest(high_n, "prompt_test_score")
    rest = scored.drop(index=high.index)
    random = rest.sample(n=min(random_n, len(rest)), random_state=random_state)
    sampled = pd.concat([high, random], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def initialize_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "dependent": "",
        "dependency_type": "",
        "relation_type": "",
        "dep_reason": "",
        "done": 0,
        "topic_shift_score": "",
        "topic_shift_flags": "",
        "topic_shift_bucket": "",
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    df["done"] = pd.to_numeric(df["done"], errors="coerce").fillna(0).astype(int)
    df["Scode"] = df["Scode"].astype(str).str.strip().str.zfill(6)
    return df


def load_or_create_candidates(args: argparse.Namespace) -> pd.DataFrame:
    output_path = Path(args.output_file)
    if args.resume and output_path.exists():
        logger.info("检测到已有输出文件，尝试续跑：%s", output_path)
        try:
            df = pd.read_csv(
                output_path,
                encoding="utf-8-sig",
                dtype={"Scode": str, "Year": str, "Q1_row": str, "Q2_row": str, "distance": str},
                low_memory=False,
            )
        except EmptyDataError:
            logger.info("输出文件为空，重新构造。")
        else:
            return initialize_columns(df)

    raw = load_data(args.input_file)
    logger.info("原始记录数：%d，session 数：%d", len(raw), raw["session_id"].nunique())
    candidates = build_candidates(raw, args.window_size)
    logger.info("构造候选边：%d", len(candidates))
    if args.min_q2_length > 0:
        before = len(candidates)
        candidates = candidates[candidates["Q2"].str.len() >= args.min_q2_length].copy()
        logger.info("Q2 长度 >= %d 后：%d/%d", args.min_q2_length, len(candidates), before)
    candidates = sample_candidates(candidates, args.max_candidates, args.random_state)
    candidates = initialize_columns(candidates)
    if args.resume:
        candidates.to_csv(args.output_file, index=False, encoding="utf-8-sig")
        logger.info("候选边初始化保存至：%s", args.output_file)
    return candidates


def apply_topic_shift_denoise(df: pd.DataFrame, easy_threshold: float, boundary_threshold: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    scores: List[float] = []
    flags: List[str] = []
    buckets: List[str] = []

    for row in out.itertuples(index=False):
        score, row_flags = score_topic_shift(getattr(row, "Q1", ""), getattr(row, "A1", ""), getattr(row, "Q2", ""))
        scores.append(score)
        flags.append("|".join(row_flags))
        if score >= easy_threshold:
            buckets.append("easy")
        elif score >= boundary_threshold:
            buckets.append("boundary")
        else:
            buckets.append("keep")

    out["topic_shift_score"] = scores
    out["topic_shift_flags"] = flags
    out["topic_shift_bucket"] = buckets

    easy_mask = out["topic_shift_bucket"] == "easy"
    if easy_mask.any():
        out.loc[easy_mask, "done"] = 1
        out.loc[easy_mask, "dependent"] = "No"
        out.loc[easy_mask, "dependency_type"] = "No-Dependency"
        out.loc[easy_mask, "relation_type"] = "Topic Shift"
        out.loc[easy_mask, "dep_reason"] = "Automatic topic shift denoise."
    return out


def parse_json_response(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    if not content:
        raise ValueError("empty response content")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_json(client: OpenAI, args: argparse.Namespace, user_content: str) -> Dict[str, Any]:
    for attempt in range(args.max_api_retries):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": FOLLOWUP_NETWORK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": args.temperature,
                "stream": False,
            }
            if args.json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            if args.enable_thinking:
                request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

            response = client.chat.completions.create(**request_kwargs)
            return parse_json_response(response.choices[0].message.content)
        except Exception as exc:
            logger.warning("API 调用失败 %d/%d: %s", attempt + 1, args.max_api_retries, exc)
            time.sleep(args.retry_sleep_seconds * (attempt + 1))
    return {"dependent": "Error", "dependency_type": "Error", "relation_type": "Error", "reason": "API failed"}


def label_one(client: OpenAI, args: argparse.Namespace, row: pd.Series) -> Dict[str, Any]:
    user_content = (
        f"Q1: {row['Q1']}\n"
        f"A1: {row['A1']}\n"
        f"Q2: {row['Q2']}\n"
        f"distance: {row['distance']}"
    )
    result = call_llm_json(client, args, user_content)
    dependent = result.get("dependent", "Error")
    dependency_type = result.get("dependency_type", "Error")
    relation_type = result.get("relation_type", "Error")
    reason = result.get("reason", str(result))

    if dependent not in {"Yes", "No"}:
        dependent = "Error"
    if relation_type == "Interaction-Complaint":
        relation_type = "Complaint"
    if dependent == "No":
        dependency_type = "No-Dependency"
        relation_type = "Topic Shift"
    return {
        "dependent": dependent,
        "dependency_type": dependency_type,
        "relation_type": relation_type,
        "dep_reason": reason,
    }


def save_progress(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """保存主标注文件。"""
    df["Scode"] = df["Scode"].astype(str).str.strip().str.zfill(6)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False, encoding="utf-8-sig")


def should_label_row(row: pd.Series, retry_errors: bool) -> bool:
    """判断当前行是否需要标注或重试。"""
    done = int(row.get("done", 0) or 0)
    dependent = str(row.get("dependent", "")).strip()
    if done != 1:
        return True
    if retry_errors and dependent == "Error":
        return True
    return False

def run(args: argparse.Namespace) -> None:
    df = load_or_create_candidates(args)
    if args.enable_topic_shift_denoise:
        df = apply_topic_shift_denoise(df, args.easy_shift_threshold, args.boundary_shift_threshold)
        logger.info(
            "Topic shift denoise enabled: easy=%d boundary=%d keep=%d",
            int((df["topic_shift_bucket"] == "easy").sum()),
            int((df["topic_shift_bucket"] == "boundary").sum()),
            int((df["topic_shift_bucket"] == "keep").sum()),
        )
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    client = create_client(args)

    total = len(df)
    done = int((df["done"] == 1).sum())
    errors = int(((df["done"] == 1) & (df["dependent"] == "Error")).sum())
    pending_mask = df.apply(lambda row: should_label_row(row, args.retry_errors), axis=1)
    pending_indices = df.index[pending_mask].tolist()
    if args.limit > 0:
        pending_indices = pending_indices[: args.limit]
    logger.info(
        "LLM config: model=%s base_url=%s reasoning_effort=%s thinking=%s json_mode=%s",
        args.model,
        args.base_url,
        args.reasoning_effort or "off",
        args.enable_thinking,
        args.json_mode,
    )
    logger.info(
        "标注候选数：%d，已完成：%d，Error：%d，本次待处理：%d",
        total,
        done,
        errors,
        len(pending_indices),
    )

    processed = 0
    consecutive_errors = 0
    try:
        for idx in pending_indices:
            row = df.loc[idx]
            if processed % args.log_every == 0:
                current_done = int((df["done"] == 1).sum())
                yes = int(((df["done"] == 1) & (df["dependent"] == "Yes")).sum())
                no = int(((df["done"] == 1) & (df["dependent"] == "No")).sum())
                current_errors = int(((df["done"] == 1) & (df["dependent"] == "Error")).sum())
                logger.info(
                    "[row=%d/%d run=%d/%d done=%d] session=%s Q1=%s Q2=%s | Yes=%d No=%d Error=%d",
                    idx + 1,
                    total,
                    processed + 1,
                    len(pending_indices),
                    current_done,
                    row["session_id"],
                    row["Q1_row"],
                    row["Q2_row"],
                    yes,
                    no,
                    current_errors,
                )

            result = label_one(client, args, row)
            for col, value in result.items():
                df.at[idx, col] = value
            df.at[idx, "done"] = 1
            processed += 1
            if result.get("dependent") == "Error":
                consecutive_errors += 1
                if args.max_consecutive_errors > 0 and consecutive_errors >= args.max_consecutive_errors:
                    save_progress(df, args)
                    logger.error(
                        "连续 API/Error 达到 %d 次，已保存进度并停止。可稍后使用同一命令续跑，或加 --retry-errors 重试 Error 行。",
                        args.max_consecutive_errors,
                    )
                    return
            else:
                consecutive_errors = 0

            if processed % args.save_every == 0:
                save_progress(df, args)
                logger.info("已保存进度：本次 processed=%d/%d", processed, len(pending_indices))

            time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在保存当前进度。")
        save_progress(df, args)
        raise
    except Exception:
        logger.exception("运行异常，正在保存当前进度。")
        save_progress(df, args)
        raise

    save_progress(df, args)

    logger.info("标注完成：%s", args.output_file)
    logger.info("dependent 分布：\n%s", df["dependent"].value_counts(dropna=False))
    logger.info("dependency_type 分布：\n%s", df["dependency_type"].value_counts(dropna=False))
    logger.info("relation_type 分布：\n%s", df["relation_type"].value_counts(dropna=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label a small candidate sample to test the follow-up network prompt.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--min-q2-length", type=int, default=0, help="0 keeps short interaction-followups.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="每处理多少条保存一次主文件。")
    parser.add_argument("--log-every", type=int, default=25, help="每处理多少条打印一次进度。")
    parser.add_argument("--limit", type=int, default=0, help="本次最多处理多少条待标注数据；0 表示不限。")
    parser.add_argument("--max-api-retries", type=int, default=DEFAULT_MAX_API_RETRIES)
    parser.add_argument("--retry-sleep-seconds", type=float, default=DEFAULT_RETRY_SLEEP_SECONDS)
    parser.add_argument("--max-consecutive-errors", type=int, default=DEFAULT_MAX_CONSECUTIVE_ERRORS, help="连续多少条 API/Error 后保存并停止；0 表示不停止。")
    parser.add_argument("--enable-topic-shift-denoise", action="store_true", default=True, help="先自动剔除明显 Topic Shift 负例。")
    parser.add_argument("--disable-topic-shift-denoise", action="store_false", dest="enable_topic_shift_denoise", help="关闭自动 Topic Shift 去噪。")
    parser.add_argument("--easy-shift-threshold", type=float, default=DEFAULT_EASY_SHIFT_THRESHOLD)
    parser.add_argument("--boundary-shift-threshold", type=float, default=DEFAULT_BOUNDARY_SHIFT_THRESHOLD)
    parser.add_argument("--retry-errors", action="store_true", help="续跑时重新标注 dependent=Error 的行。")
    parser.add_argument("--model", default=MODEL_NAME, help="DeepSeek model name, e.g. deepseek-v4-pro or deepseek-chat.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", ""], help="DeepSeek V4 reasoning effort. Use empty string to disable.")
    parser.add_argument("--enable-thinking", action="store_true", help="Send extra_body={thinking:{type:enabled}} for DeepSeek V4.")
    parser.add_argument("--no-json-mode", action="store_true", help="Disable response_format json_object if the target model does not support it.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.resume = not args.no_resume
    args.json_mode = not args.no_json_mode
    if args.reasoning_effort == "":
        args.reasoning_effort = None
    return args


if __name__ == "__main__":
    run(parse_args())
