# -*- coding: utf-8 -*-
"""Shared semantic constraints and prototype helpers for the 0630 pipeline.

This module is the single source of truth for semantic constraints used by the
0630 workflow. It intentionally centralizes:

- reason category patterns
- reason prototype prompts
- topic-shift and pressure lexicons
- helper functions for semantic bridge construction

Keep semantic edits here when possible so Stage 1-2, targeted data generation,
and MNRL fine-tuning stay aligned.

Design notes:
- Keep the prototypes short and stable so anchors stay readable.
- Prefer the more specific business categories first; use answer-pressure as a
  fallback when the explanation mainly expresses direct pressure or
  clarification rather than a precise business topic.
- If you need to change the semantic contract of the 0630 workflow, change it
  here first and let 1/2/3 inherit the update automatically.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd

# =============================================================================
# Reason categories and their compact prototype prompts
# =============================================================================

# Pattern lexicon for mapping explanation text to compact reason categories.
# The categories are ordered from more specific business semantics to the more
# generic pressure / clarification fallback. The first match wins.
REASON_CATEGORY_PATTERNS = {
    # Public disclosure / detail-followup semantics.
    "disclosure_detail": (
        "披露",
        "公告",
        "年报",
        "季报",
        "回避",
        "细节",
        "标准",
        "程序",
        "进展",
        "说明",
    ),
    # Shareholder return / dividend / buyback semantics.
    "shareholder_return": (
        "股东回报",
        "分红",
        "派息",
        "减持",
        "回购",
        "送转",
        "不减持",
    ),
    # Restructuring / capital injection / financing semantics.
    "restructuring": (
        "重组",
        "增发",
        "资产注入",
        "非公开发行",
        "并购",
        "障碍",
        "限制",
        "时间表",
    ),
    # Strategic planning / business direction semantics.
    "strategy_planning": (
        "规划",
        "战略",
        "主业",
        "发展方向",
        "定位",
        "未来",
        "目标",
        "布局",
    ),
    # Performance / governance / management change semantics.
    "performance_governance": (
        "业绩",
        "利润",
        "盈利",
        "成本",
        "高管",
        "辞职",
        "治理",
        "经营",
        "减亏",
    ),
    # Direct pressure / clarification semantics. This is the broadest category
    # and should behave as a fallback when no more specific business category
    # is clearly indicated.
    "answer_pressure": (
        "明确回答",
        "明确的回答",
        "给个明确",
        "请回答",
        "正面回答",
        "请正面回答",
        "不要回避",
        "不明确",
        "笼统",
        "施压",
        "催促",
        "务必重视",
        "明确",
    ),
}

# Short, stable prototypes injected into anchors / prompts.
# Keep these concise and category-specific so they act like semantic anchors,
# not long explanations.
# REASON_CATEGORY_PROMPTS = {
#     "disclosure_detail": "【理由类别:披露细节】对笼统回答继续追问具体进展、量化结果、时间表、审批程序和落地执行细节。",
#     "answer_pressure": "【理由类别:答复施压】对笼统、回避或感性回应继续施压，要求正面回答、给出明确时间、具体计划和可验证的执行细节。",
#     "shareholder_return": "【理由类别:股东回报】继续追问分红、派息、回购以及回报承诺的兑现时间和具体形式。",
#     "strategy_planning": "【理由类别:经营规划】继续追问经营规划、战略方向、主业布局、未来目标和发展路线。",
#     "restructuring": "【理由类别:重组】继续追问重组动向、增发进展、资产注入、障碍和时间表。",
#     "performance_governance": "【理由类别:业绩治理】继续追问业绩变化、利润波动、高管辞职、治理责任和改善路径。",
# }

REASON_CATEGORY_PROMPTS = {
    "disclosure_detail": "【理由类别:披露细节】继续追问具体进展、量化结果、时间表和落地执行细节。",
    "answer_pressure": "【理由类别:答复施压】要求正面回答、明确时间、具体计划和可验证细节。",
    "shareholder_return": "【理由类别:股东回报】继续追问分红、派息、回购和兑现时间。",
    "strategy_planning": "【理由类别:经营规划】继续追问经营规划、战略方向、主业布局和未来目标。",
    "restructuring": "【理由类别:重组】继续追问重组动向、增发进展、资产注入和时间表。",
    "performance_governance": "【理由类别:业绩治理】继续追问业绩变化、利润波动、高管辞职和治理责任。",
}

# High-confidence Topic Shift markers used by the denoiser.
# These are intentionally conservative and only meant to catch obvious topic
# changes so easy negatives can be removed without touching positives.
TOPIC_SHIFT_PATTERNS = (
    "换个话题",
    "另一个问题",
    "无关",
    "没有关系",
    "跑题",
    "偏题",
    "答非所问",
)

# A1 response-state markers used by Stage 1.
# These indicate that the previous answer is incomplete, evasive, or only a
# partial acknowledgment, which is useful for follow-up detection.
ANSWER_UNSATISFIED_TERMS = [
    "感谢关注",
    "谢谢",
    "请关注",
    "关注公告",
    "以公告为准",
    "以公司公告为准",
    "详见公告",
    "后续公告",
    "及时公告",
    "信息披露",
    "不便评价",
    "无法评价",
    "无法判断",
    "按规定",
    "按要求",
    "会努力",
    "努力工作",
    "持续关注",
]

# Q2 pressure / challenge markers used by Stage 1.
# These flag follow-up questions that explicitly challenge or demand
# clarification from the prior answer.
Q2_PRESSURE_TERMS = [
    "请正面回答",
    "正面回答",
    "明确回答",
    "请回答",
    "为什么不回答",
    "为何不回答",
    "为何不披露",
    "为什么不披露",
    "绕弯子",
    "言之无物",
    "外交辞令",
    "敷衍",
    "漠视",
    "知情权",
    "隐瞒",
    "忽悠",
    "欺骗",
    "遮遮掩掩",
    "到底",
    "问责",
    "解释一下",
    "给明确回答",
    "给个明确",
    "给个明确的回答",
    "明确的回答",
    "请明确",
    "作解释",
    "请作解释",
    "作下解释",
    "下解释",
    "作出解释",
    "做出解释",
    "请您明确回答",
    "回答我的两个问题",
    "请回答我的两个问题",
    "正面回应",
    "请正面回应",
    "直面回答",
    "从不直面回答",
    "不要回避",
    "明确目标",
    "明确的目标",
]

# Q2 explanation-only markers used by Stage 1.
# These are softer clarification cues and help separate plain explanation
# requests from harder challenge / pressure follow-ups.
Q2_EXPLAIN_ONLY_TERMS = [
    "解释一下",
    "说明一下",
    "请说明",
    "说一下",
    "介绍一下",
    "讲一下",
    "具体说说",
    "具体说明",
    "具体介绍",
    "详细说明",
    "详细介绍",
    "详细讲讲",
    "具体讲讲",
    "请讲",
    "请介绍",
]

# Ordered columns to inspect when inferring reason categories from a row.
# We prefer explanation-like columns first so LLM or human rationale wins over
# noisier fallback fields when multiple columns are present.
REASON_TEXT_COLUMNS: Tuple[str, ...] = (
    "thought_process",
    "dep_reason",
    "llm_reason",
    "explanation",
    "reason_explanation",
    "reason_text",
    "followup_reason",
    "reason_category",
)

# Backward-compatible alias for scripts that still refer to this name.
REASON_EXPLANATION_COLUMNS: Tuple[str, ...] = REASON_TEXT_COLUMNS


def clean_text(value: object) -> str:
    """Normalize text-like values to a single-space string."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def get_reason_prompt(reason_category: str) -> str:
    """Return the short prototype prompt for a semantic reason category."""
    return REASON_CATEGORY_PROMPTS.get(reason_category, "")


def infer_reason_category(reason_text: object) -> str:
    """Infer a compact reason category from a free-form explanation string."""
    text = clean_text(reason_text)
    if not text:
        return ""
    for category, patterns in REASON_CATEGORY_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return category
    return ""


def infer_reason_category_from_row(row: object, columns: Sequence[str] = REASON_TEXT_COLUMNS) -> Tuple[str, str, str]:
    """Return (category, source_field, source_text) from the first matching column."""
    for field in columns:
        value = clean_text(getattr(row, field, ""))
        if not value:
            continue
        category = infer_reason_category(value)
        if category:
            return category, field, value
    return "", "", ""


def build_reason_bridge(reason_text: object) -> str:
    """Build a reason prototype bridge from raw explanation text."""
    category = infer_reason_category(reason_text)
    prompt = get_reason_prompt(category)
    return (" " + prompt) if prompt else ""


def build_reason_bridge_for_category(reason_category: str) -> str:
    """Build a reason prototype bridge directly from a known category."""
    prompt = get_reason_prompt(reason_category)
    return (" " + prompt) if prompt else ""


def build_semantic_bridge(a1_text: str) -> str:
    """Build a lightweight lexical semantic bridge from A1 text.

    This is a lexical fallback only. If a reason prototype bridge already
    exists, pipeline callers should suppress this one to avoid feature stacking.
    """
    if not a1_text:
        return ""
    if any(term in a1_text for term in ("披露", "公告", "年报", "季报", "说明", "回避", "细节")):
        return "（注：这里通常会延展到披露细节、回避后补充说明、具体进展、时间表或审批细节。）"
    if any(term in a1_text for term in ("股民", "股东", "投资者")) and "回报" in a1_text:
        return "（注：这里的回报通常可延展为分红、派息、送转、回购、减持承诺兑现等具体事项。）"
    if any(term in a1_text for term in ("重组", "增发", "资产注入", "非公开发行")):
        return "（注：这里的业务线通常会延展到重组动向、资产注入、增发进展、审批限制、障碍和时间表。）"
    if any(term in a1_text for term in ("发展", "规划", "战略", "主业", "前景", "定位")):
        return "（注：这里的业务线通常会延展到经营规划、战略方向、主业布局、未来目标和下一步路线。）"
    if any(term in a1_text for term in ("业绩", "利润", "盈利", "成本", "利润率", "亏损", "高管", "辞职", "治理")):
        return "（注：这里的业务线通常会延展到业绩变化、盈利能力、成本压力、高管辞职、治理责任和改善路径。）"
    return ""
