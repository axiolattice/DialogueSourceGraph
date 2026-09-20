# -*- coding: utf-8 -*-
"""Final standalone FCN Stage 1.

Frozen protocol
---------------
- Train:      176 sessions
- Validation:  44 sessions
- Test:        55 sessions
- Session assignment: mandatory external manifest shared with encoder training
- Stage 1 scorer: StandardScaler + class-balanced Logistic Regression
- Scorer fitting: Train only, exactly once
- Retention selection: Validation only
- Target validation candidate-edge recall: 0.98
- Search: source-wise Top-K in {1,2,3,4,5}; threshold selected exactly from
  validation scores
- Selection criterion: minimize retained validation candidates subject to
  Recall >= 0.98
- No scorer refit after validation
- Test labels are never used for fitting or parameter selection
- The encoder provenance must certify Train-only positive-pair MNRL with no
  rationale, targeted, hard-negative, or direction-auxiliary supervision

Outputs for Stage 2A
--------------------
- stage1_train_scored.csv
- stage1_validation_scored.csv
- stage1_test_scored.csv

This file is standalone with respect to FCN Stage 1 logic: it does not import
the earlier revised/sensitivity Stage 1 scripts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import logging
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOGGER = logging.getLogger("fcn-stage1-retention")
SEED = 42

# Frozen final Stage 1 protocol.
TARGET_VALIDATION_RECALL = 0.98
MAX_TOP_K = 5
EXPECTED_TRAIN_SESSIONS = 176
EXPECTED_VALIDATION_SESSIONS = 44
EXPECTED_TEST_SESSIONS = 55
DEFAULT_DATA_FILE = PROJECT_DIR / "train data" / "fcn_30firms_full_labeled.csv"
DEFAULT_SESSION_MANIFEST = PROJECT_DIR / "train data" / "session_split_manifest.csv"
DEFAULT_MODEL_DIR = PROJECT_DIR / "checkpoints" / "fcn-m3e-base-train-only"
DEFAULT_STAGE1_OUTDIR = PROJECT_DIR / "fcn_outputs_stage1"
INPUT_FORMAT = "raw_q1_a1__q2_v1"


def clean_text(value: object) -> str:
    """Convert a value to normalized, single-space text."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return " ".join(str(value).split())


def build_anchor_text(q1: object, a1: object) -> str:
    """Build the encoder input for a candidate source turn."""
    return f"前序问题：{clean_text(q1)} 管理层回答：{clean_text(a1)}"


def build_candidate_text(q2: object) -> str:
    """Build the encoder input for a later query."""
    return f"当前追问：{clean_text(q2)}"


def text_protocol_fingerprint() -> str:
    """Hash the executable text-normalization/template contract."""
    normalized_functions = []
    for function in (clean_text, build_anchor_text, build_candidate_text):
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        # Documentation changes do not alter the executable protocol.
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        normalized_functions.append(ast.dump(node, include_attributes=False))
    payload = "\n".join(normalized_functions).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


TEXT_PROTOCOL_SHA256 = text_protocol_fingerprint()


# Stage 1-only fixed interaction lexicons.
ANSWER_UNSATISFIED_TERMS = [
    "感谢关注", "谢谢", "请关注", "关注公告", "以公告为准", "以公司公告为准",
    "详见公告", "后续公告", "及时公告", "信息披露", "不便评价", "无法评价",
    "无法判断", "按规定", "按要求", "会努力", "努力工作", "持续关注",
]
Q2_PRESSURE_TERMS = [
    "请正面回答", "正面回答", "明确回答", "请回答", "为什么不回答", "为何不回答",
    "为何不披露", "为什么不披露", "绕弯子", "言之无物", "外交辞令", "敷衍",
    "漠视", "知情权", "隐瞒", "忽悠", "欺骗", "遮遮掩掩", "到底", "问责",
    "解释一下", "给明确回答", "给个明确", "给个明确的回答", "明确的回答",
    "请明确", "作解释", "请作解释", "作下解释", "下解释", "作出解释",
    "做出解释", "请您明确回答", "回答我的两个问题", "请回答我的两个问题",
    "正面回应", "请正面回应", "直面回答", "从不直面回答", "不要回避",
    "明确目标", "明确的目标",
]
Q2_EXPLAIN_ONLY_TERMS = [
    "解释一下", "说明一下", "请说明", "说一下", "介绍一下", "讲一下", "具体说说",
    "具体说明", "具体介绍", "详细说明", "详细介绍", "详细讲讲", "具体讲讲",
    "请讲", "请介绍",
]
A1_CLAIM_TERMS = [
    "因为",
    "由于",
    "主要原因",
    "原因",
    "预计",
    "将会",
    "正在",
    "已经",
    "不存在",
    "不会影响",
    "公司认为",
    "公司将",
    "我们会",
    "正在推进",
    "进展顺利",
    "积极推进",
    "及时披露",
    "严格按照",
]
Q2_CHALLENGE_TERMS = [
    "为何",
    "为什么",
    "难道",
    "但是",
    "可是",
    "实际上",
    "是否说明",
    "是否存在",
    "是不是",
    "怎么解释",
    "解释原因",
    "原因是什么",
    "出了什么问题",
    "不作披露",
    "不披露",
    "拖延",
    "承诺",
    "兑现",
    "质疑",
]
EVIDENCE_TERMS = [
    "公告",
    "媒体",
    "报道",
    "数据显示",
    "年报",
    "一季报",
    "半年报",
    "财报",
    "证监会",
    "交易所",
    "药监局",
    "批件",
    "临床",
    "项目延期",
    "亏损",
    "减值",
    "处罚",
    "诉讼",
]

BUSINESS_EVENT_GROUPS: Dict[str, Sequence[str]] = {
    "project_capacity": [
        "募投",
        "募集资金",
        "项目",
        "投产",
        "达产",
        "产能",
        "扩产",
        "生产线",
        "建设",
        "开业",
        "落地",
        "推进",
        "进展",
    ],
    "performance_growth": [
        "收入",
        "营收",
        "销售",
        "利润",
        "净利润",
        "业绩",
        "增长",
        "成长",
        "毛利",
        "毛利率",
        "贡献",
        "盈利",
        "效益",
    ],
    "market_order_price": [
        "订单",
        "市场",
        "客户",
        "销售",
        "价格",
        "涨价",
        "降价",
        "需求",
        "竞争",
        "份额",
        "渠道",
        "推广",
    ],
    "cost_margin": [
        "成本",
        "原材料",
        "毛利",
        "毛利率",
        "费用",
        "管理费用",
        "营业费用",
        "运输费用",
        "价格",
        "涨价",
        "亏损",
        "一季度亏损",
        "减值",
        "盈利能力",
    ],
    "product_rnd": [
        "产品",
        "新品",
        "研发",
        "临床",
        "批件",
        "审批",
        "药品",
        "制剂",
        "医疗制剂",
        "医疗机构制剂",
        "院内制剂",
        "技术",
        "商业化",
        "医院",
        "执业许可证",
        "临床试验",
    ],
    "disclosure_process": [
        "公告",
        "披露",
        "证监会",
        "交易所",
        "股东大会",
        "审批",
        "核准",
        "上报",
        "年报",
        "一季报",
        "半年报",
        "进度",
    ],
    "capital_restructuring": [
        "重组",
        "资产重组",
        "非公开发行",
        "收购",
        "股权",
        "股东",
        "增发",
        "定增",
        "注入",
        "购买资产",
        "资产购买",
        "借壳",
        "不重组",
        "限制",
        "障碍",
    ],
    "development_strategy": [
        "发展规划",
        "发展方向",
        "发展",
        "规划",
        "目标",
        "方向",
        "前景",
        "主业",
        "多元化",
        "战略",
        "转型",
        "投资建设",
        "对外投资",
        "项目规划",
    ],
    "shareholder_return": [
        "分红",
        "送股",
        "送转",
        "回报",
        "市值",
        "减持",
        "承诺",
        "提案",
        "股东",
    ],
    "management_governance": [
        "高管",
        "辞职",
        "管理层",
        "董事",
        "董事会",
        "股东",
        "散户",
        "减持",
        "投资行为",
    ],
}

Q2_INTENT_GROUPS: Dict[str, Sequence[str]] = {
    "growth_profit_question": [
        "成长性",
        "增长点",
        "利润",
        "盈利",
        "业绩",
        "收入",
        "营收",
        "效益",
        "贡献",
        "每股收益",
        "毛利",
        "毛利率",
        "回报",
    ],
    "progress_detail_question": [
        "进展",
        "进度",
        "什么时候",
        "何时",
        "是否顺利",
        "是否完成",
        "能否",
        "具体",
        "哪些",
        "多少",
        "程序",
        "审批",
        "上报",
        "核准",
    ],
    "cause_explanation_question": [
        "为什么",
        "为何",
        "原因",
        "怎么解释",
        "是否因为",
        "是否说明",
        "影响",
        "困难",
        "障碍",
        "问题",
        "亏损",
        "费用",
        "成本",
    ],
    "disclosure_detail_question": [
        "披露",
        "公告",
        "知情权",
        "详细",
        "具体",
        "数据",
        "金额",
        "合同",
        "进展",
        "是否属实",
    ],
    "commitment_challenge_question": [
        "承诺",
        "兑现",
        "是否属实",
        "是不是",
        "难道",
        "笔误",
        "矛盾",
        "拖延",
        "不披露",
        "质疑",
    ],
    "planning_question": [
        "发展",
        "规划",
        "发展规划",
        "发展方向",
        "打算",
        "目标",
        "方向",
        "前景",
        "未来",
        "下一步",
        "计划",
        "好处",
        "态度",
    ],
    "capital_transaction_question": [
        "重组",
        "非公开发行",
        "购买资产",
        "资产购买",
        "限制",
        "障碍",
        "区别",
        "好处",
        "借壳",
        "完成",
        "不重组",
    ],
    "shareholder_return_question": [
        "分红",
        "送股",
        "送转",
        "回报",
        "市值",
        "承诺",
        "减持",
        "提案",
        "信心",
    ],
    "product_plan_question": [
        "规划",
        "定位",
        "目标",
        "产品",
        "申请",
        "临床试验",
        "暂停",
        "继续投入",
        "开业",
        "许可证",
        "制剂",
    ],
}

BUSINESS_BRIDGE_RULES: Sequence[Tuple[str, str, float]] = [
    ("project_capacity", "growth_profit_question", 0.28),
    ("project_capacity", "progress_detail_question", 0.30),
    ("performance_growth", "growth_profit_question", 0.22),
    ("performance_growth", "cause_explanation_question", 0.24),
    ("market_order_price", "growth_profit_question", 0.22),
    ("market_order_price", "cause_explanation_question", 0.24),
    ("cost_margin", "growth_profit_question", 0.22),
    ("cost_margin", "cause_explanation_question", 0.26),
    ("product_rnd", "growth_profit_question", 0.20),
    ("product_rnd", "progress_detail_question", 0.28),
    ("disclosure_process", "progress_detail_question", 0.28),
    ("disclosure_process", "disclosure_detail_question", 0.30),
    ("disclosure_process", "capital_transaction_question", 0.26),
    ("capital_restructuring", "progress_detail_question", 0.32),
    ("capital_restructuring", "commitment_challenge_question", 0.30),
    ("capital_restructuring", "growth_profit_question", 0.26),
    ("capital_restructuring", "planning_question", 0.28),
    ("capital_restructuring", "capital_transaction_question", 0.34),
    ("development_strategy", "planning_question", 0.30),
    ("development_strategy", "growth_profit_question", 0.22),
    ("market_order_price", "planning_question", 0.20),
    ("cost_margin", "planning_question", 0.20),
    ("shareholder_return", "shareholder_return_question", 0.32),
    ("shareholder_return", "commitment_challenge_question", 0.26),
    ("management_governance", "commitment_challenge_question", 0.28),
    ("management_governance", "planning_question", 0.22),
    ("product_rnd", "product_plan_question", 0.32),
    ("product_rnd", "planning_question", 0.24),
]

STRUCTURED_FOLLOWUP_ACTION_TERMS = [
    "为什么",
    "为何",
    "原因",
    "怎么解释",
    "具体",
    "详细",
    "多少",
    "哪些",
    "何时",
    "什么时候",
    "是否",
    "能否",
    "有没有",
    "怎么样",
    "如何",
    "进展",
    "进度",
    "影响",
    "贡献",
    "带来",
    "体现",
    "正常",
    "异常",
    "改善",
    "措施",
    "计划",
    "目标",
    "方向",
    "规划",
    "打算",
    "态度",
    "建议",
    "障碍",
    "限制",
    "承诺",
    "兑现",
    "属实",
]

STRUCTURED_REFERENCE_TERMS = [
    "上述",
    "前述",
    "刚才",
    "该",
    "这个",
    "这些",
    "此",
    "对此",
    "基于",
    "既然",
]

A1_ANSWER_STATE_TERMS = [
    "尚未",
    "未",
    "没有",
    "不确定",
    "存在不确定性",
    "正在",
    "将会",
    "会",
    "计划",
    "预计",
    "努力",
    "有信心",
    "有希望",
    "需",
    "需要",
    "须",
    "办理",
    "办理中",
    "审批",
    "批准",
    "核准",
    "推进",
    "进展",
    "及时披露",
    "后续公告",
    "关注公告",
    "以公告为准",
]

A1_COMMITMENT_STATE_TERMS = [
    "承诺",
    "不进行",
    "不会",
    "没有",
    "不存在",
    "严格按照",
    "及时履行",
    "及时披露",
    "目标",
    "规划",
    "方向",
]

Q2_CONTINUATION_NEED_TERMS = [
    "为什么",
    "为何",
    "原因",
    "怎么解释",
    "是否因为",
    "是否说明",
    "进展",
    "进度",
    "何时",
    "什么时候",
    "是否完成",
    "能否",
    "是否",
    "具体",
    "详细",
    "多少",
    "哪些",
    "如何",
    "影响",
    "好处",
    "限制",
    "障碍",
    "困难",
    "态度",
    "规划",
    "目标",
    "方向",
    "申请",
    "临床试验",
    "继续投入",
    "暂停",
    "开业",
    "兑现",
    "承诺",
]

STRUCTURED_CONCRETE_OBJECT_TERMS = [
    "智能卡",
    "智能卡基材",
    "基材",
    "PVC",
    "pvc",
    "原油",
    "石油",
    "树脂",
    "品级",
    "汇率",
    "扇贝",
    "虾夷扇贝",
    "浮筏",
    "海参",
    "鲍鱼",
    "棉花",
    "天竹纤维",
    "莫代尔",
    "天丝纤维",
    "重组",
    "重大资产重组",
    "重组方案",
    "重组完成",
    "立案调查",
    "结案",
    "批准",
    "证监会批准",
    "中国证监会",
    "摆脱困境",
    "非公开发行",
    "非公开发行股票",
    "购买资产",
    "股票购买资产",
    "增发",
    "定增",
    "不重组",
    "六个月",
    "6个月",
    "限制",
    "障碍",
    "减持",
    "承诺",
    "高管",
    "辞职",
    "渠道",
    "宁波",
    "分红",
    "送股",
    "现金流",
    "负债",
    "融资",
    "抵押",
    "海冰",
    "自然灾害",
    "风险控制",
    "战略转型",
    "调味品",
    "蚝油",
    "盐碱地",
    "土壤改良",
    "PPP",
    "通辽",
    "燃油",
    "燃油价格",
    "石油涨价",
    "原材料价格",
    "运输费用",
    "管理费用",
    "营业费用",
    "费用增加",
    "产品利润率",
    "总收益",
    "一季度亏损",
    "亏损",
    "散户",
    "中小散户",
    "煤炭",
    "煤炭贸易",
    "资产注入",
    "优质资产",
    "收购优质资产",
    "细分市场",
    "市场占有率",
    "智慧城市",
    "智能建筑",
    "世博会",
    "世博",
    "合同",
    "合同规模",
    "送转",
    "回报股东",
    "太子参",
    "党参",
    "胶原蛋白",
    "爱透",
    "火透",
    "凉茶",
    "互联网医疗",
    "大数据",
    "大健康",
    "糖尿病医院",
    "长沙医院",
    "301医院",
    "替芬泰",
    "Y101",
    "临床批件",
    "腾讯",
    "慢性病中心",
    "卫计委",
    "政府资金",
    "补贴",
    "苗药",
    "糖宁通络",
    "院内制剂",
    "医疗制剂",
    "黑笔事件",
    "高管辞职",
    "管理层",
    "房产公司",
    "房地产投资",
    "江苏国信",
    "国信资产",
    "江苏省房地产",
    "借壳",
    "专心",
    "发展规划",
    "发展方向",
    "明确方向",
    "项目规划",
    "投资建设",
    "对外投资",
    "下一步发展",
    "产品规划",
    "产品定位",
    "凉茶产品",
    "医疗机构制剂",
    "执业许可证",
    "临床试验",
    "糖尿病",
    "糖尿病医院",
    "虎耳草",
    "山银花",
    "云南白药",
    "合作项目",
]

STRUCTURED_TERM_STOPWORDS = {
    "公司",
    "请问",
    "您好",
    "谢谢",
    "贵公司",
    "董事长",
    "投资者",
    "股东",
    "问题",
    "回答",
    "情况",
    "目前",
    "未来",
    "方面",
    "具体",
    "是否",
    "能否",
    "为什么",
    "原因",
    "如何",
    "哪些",
    "多少",
    "什么",
    "项目",
    "业务",
    "产品",
    "市场",
    "同行业",
    "发展",
    "经营",
    "正常",
    "投资",
    "收益",
    "盈利",
    "能力",
    "盈利能力",
    "增长",
    "业绩",
    "第一",
    "主要",
    "公司的",
    "公司有",
    "有限公司",
    "股份有限公司",
    "公告",
    "披露",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_split(value: object) -> str:
    split = clean_text(value).lower()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    return aliases.get(split, split)


def load_session_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Session manifest does not exist: {path}")
    manifest = pd.read_csv(path, encoding="utf-8-sig", dtype={"session_id": str})
    required = {"session_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Session manifest missing columns: {sorted(missing)}")
    manifest = manifest.loc[:, ["session_id", "split"]].copy()
    manifest["session_id"] = manifest["session_id"].map(clean_text)
    manifest["split"] = manifest["split"].map(normalize_split)
    if (manifest["session_id"] == "").any():
        raise ValueError("Session manifest contains an empty session_id.")
    duplicates = manifest.loc[manifest["session_id"].duplicated(False), "session_id"].unique()
    if len(duplicates):
        raise ValueError(f"Session manifest contains duplicate session IDs: {duplicates[:10].tolist()}")
    expected_splits = {"train", "validation", "test"}
    if set(manifest["split"]) != expected_splits:
        raise ValueError(
            f"Manifest splits must be exactly {sorted(expected_splits)}; "
            f"observed {sorted(set(manifest['split']))}."
        )
    observed_counts = manifest.groupby("split")["session_id"].nunique().to_dict()
    expected_counts = {
        "train": EXPECTED_TRAIN_SESSIONS,
        "validation": EXPECTED_VALIDATION_SESSIONS,
        "test": EXPECTED_TEST_SESSIONS,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Manifest session counts mismatch: observed={observed_counts}, "
            f"expected={expected_counts}."
        )
    return manifest


def verify_encoder_provenance(
    model_dir: Path,
    data_path: Path,
    session_manifest_path: Path,
) -> Tuple[Path, Dict[str, object]]:
    """Reject any encoder that was not trained under the formal protocol."""
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Formal encoder directory does not exist: {model_dir}")
    provenance_path = model_dir / "encoder_provenance_manifest.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            "Formal encoder provenance manifest is missing: "
            f"{provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required_values = {
        "formal_fcn_encoder": True,
        "training_objective": "positive_pair_mnrl_only",
        "input_format": INPUT_FORMAT,
        "text_protocol_sha256": TEXT_PROTOCOL_SHA256,
        "rationale_derived_supervision": False,
        "targeted_supervision": False,
        "hard_negative_supervision": False,
        "direction_auxiliary_supervision": False,
        "train_only": True,
    }
    mismatches = {
        key: {"observed": provenance.get(key), "required": value}
        for key, value in required_values.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Encoder provenance violates the formal protocol: {mismatches}")

    data_hash = sha256_file(data_path)
    manifest_hash = sha256_file(session_manifest_path)
    if provenance.get("train_file_sha256") != data_hash:
        raise RuntimeError("Encoder was trained from a different candidate-data file.")
    if provenance.get("session_manifest_sha256") != manifest_hash:
        raise RuntimeError("Encoder was trained from a different session manifest.")
    expected_counts = {
        "train": EXPECTED_TRAIN_SESSIONS,
        "validation": EXPECTED_VALIDATION_SESSIONS,
        "test": EXPECTED_TEST_SESSIONS,
    }
    if provenance.get("session_counts") != expected_counts:
        raise RuntimeError(
            "Encoder provenance has unexpected session counts: "
            f"{provenance.get('session_counts')}"
        )
    return model_dir, provenance


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    out = {
        "total": int(len(y_true_arr)),
        "true_edges": int(y_true_arr.sum()),
        "pred_edges": int(y_pred_arr.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
    }
    if y_score is not None:
        y_score_arr = np.asarray(y_score, dtype=float)
        out["auc"] = safe_auc(y_true_arr, y_score_arr)
        out["ap"] = safe_ap(y_true_arr, y_score_arr)
    return out


def print_block(title: str, metrics: Dict[str, object]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value):
                print(f"{key}: nan")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary recovery label and a legacy post-hoc metadata column.

    ``edge_label`` is the only prediction target used by this Stage 1 script.
    ``relation_type_train`` is retained solely for backward-compatible exports
    and downstream post-hoc analysis; it is not used by the Stage 1 scorer or
    retention policy.
    """
    out = df.copy()
    out["edge_label"] = (out["dependent"] == "Yes").astype(int)
    out["relation_type_train"] = out["relation_type"].replace({"Clarification": "Elaboration"})
    return out


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q1_len"] = out["Q1"].map(lambda x: len(clean_text(x)))
    out["a1_len"] = out["A1"].map(lambda x: len(clean_text(x)))
    out["q2_len"] = out["Q2"].map(lambda x: len(clean_text(x)))
    out["distance_num"] = pd.to_numeric(out["distance"], errors="coerce").fillna(0).astype(float)
    return out


def encode_embeddings(
    df: pd.DataFrame,
    model_dir: str,
    cache_dir: Path,
    batch_size: int,
    data_sha256: str,
    encoder_provenance_sha256: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode anchors and Q2 candidates using the fixed experiment input format."""
    required = {"Q1", "A1", "Q2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Stage 1 text columns: {sorted(missing)}")

    model_path = str(Path(model_dir).expanduser().resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    anchor_cache = cache_dir / "anchor_embeddings.npy"
    cand_cache = cache_dir / "candidate_embeddings.npy"
    meta_cache = cache_dir / "embedding_meta.json"
    meta = {
        "rows": len(df),
        "model_id": model_dir,
        "model_path": model_path,
        "input_format": INPUT_FORMAT,
        "text_protocol_sha256": TEXT_PROTOCOL_SHA256,
        "data_sha256": data_sha256,
        "encoder_provenance_sha256": encoder_provenance_sha256,
        "first_edge": str(df.iloc[0]["edge_id"]) if len(df) else "",
        "last_edge": str(df.iloc[-1]["edge_id"]) if len(df) else "",
    }
    if anchor_cache.exists() and cand_cache.exists() and meta_cache.exists():
        try:
            old_meta = json.loads(meta_cache.read_text(encoding="utf-8"))
            if old_meta == meta:
                LOGGER.info("Loading cached embeddings from %s", cache_dir)
                return np.load(anchor_cache), np.load(cand_cache)
        except Exception:
            pass

    LOGGER.info("Encoding texts with %s", model_path)
    model = SentenceTransformer(model_path)
    anchors = [build_anchor_text(r.Q1, r.A1) for r in df.itertuples(index=False)]
    cands = [build_candidate_text(value) for value in df["Q2"].tolist()]
    anchor_emb = model.encode(
        anchors,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    cand_emb = model.encode(
        cands,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(anchor_cache, anchor_emb)
    np.save(cand_cache, cand_emb)
    meta_cache.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.asarray(anchor_emb, dtype=np.float32), np.asarray(cand_emb, dtype=np.float32)


def build_features(df: pd.DataFrame, anchor_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
    """Construct the supervised Stage 1 edge-scoring feature vector.

    The Stage 1 scorer is not cosine-only. It combines cosine similarity,
    temporal distance, text-length features, absolute embedding differences,
    and element-wise embedding products. The fitted logistic model converts
    these features into ``stage1_semantic_prob``.
    """
    cos = np.sum(anchor_emb * cand_emb, axis=1, keepdims=True)
    diff = np.abs(anchor_emb - cand_emb)
    prod = anchor_emb * cand_emb
    numeric_cols = [
        "distance_num",
        "q1_len",
        "a1_len",
        "q2_len",
    ]
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)
    return np.hstack([cos.astype(np.float32), numeric, diff.astype(np.float32), prod.astype(np.float32)])


def term_score(text: object, terms: Sequence[str], normalizer: float) -> float:
    value = clean_text(text)
    if not value:
        return 0.0
    count = 0
    for term in terms:
        if term and term in value:
            count += 1
    return float(min(count / normalizer, 1.0))


def group_score(text: object, terms: Sequence[str]) -> float:
    value = clean_text(text)
    if not value:
        return 0.0
    hits = sum(1 for term in terms if term and term in value)
    return float(min(hits / 2.0, 1.0))


def extract_structured_terms(text: object) -> set[str]:
    value = clean_text(text)
    if not value:
        return set()
    value = re.sub(r"请问[^，。；;：:\s]{1,8}[：:]", " ", value)
    value = re.sub(r"请问[^，。；;：:\s]{1,6}", " ", value)

    terms: set[str] = set()
    for group_terms in list(BUSINESS_EVENT_GROUPS.values()) + [STRUCTURED_CONCRETE_OBJECT_TERMS]:
        for term in group_terms:
            if len(term) >= 2 and term not in STRUCTURED_TERM_STOPWORDS and term in value:
                terms.add(term)

    for chunk in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", chunk):
            if len(chunk) >= 2:
                terms.add(chunk.lower())
    return terms


def shared_object_score(anchor_text: object, q2_text: object) -> Tuple[float, str]:
    anchor_terms = extract_structured_terms(anchor_text)
    q2_terms = extract_structured_terms(q2_text)
    shared = sorted(anchor_terms & q2_terms, key=lambda term: (-len(term), term))
    if not shared:
        return 0.0, ""
    strong = [term for term in shared if len(term) >= 3]
    score = min((len(strong) * 0.5) + (len(shared) * 0.12), 1.0)
    return float(score), "|".join(shared[:12])


def compute_business_bridge_scores(
    df: pd.DataFrame,
    semantic: np.ndarray,
    near_context: np.ndarray,
    score_scale: float,
    min_semantic: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute structured bridge scores with concrete-object dependency checks."""
    bridge_scores = np.zeros(len(df), dtype=float)
    detail: Dict[str, List[float]] = {}
    for name in BUSINESS_EVENT_GROUPS:
        detail[f"a1_event_{name}"] = []
    for name in Q2_INTENT_GROUPS:
        detail[f"q2_intent_{name}"] = []
    detail["structured_followup_action_score"] = []
    detail["structured_reference_score"] = []
    detail["structured_shared_object_score"] = []
    detail["structured_a1_shared_object_score"] = []
    detail["structured_bridge_rule_score"] = []
    detail["structured_capacity_growth_signal"] = []
    detail["structured_answer_state_score"] = []
    detail["structured_answer_commitment_score"] = []
    detail["structured_answer_numeric_score"] = []
    detail["structured_q2_continuation_need_score"] = []
    shared_terms: List[str] = []
    a1_shared_terms: List[str] = []

    for row in df.itertuples(index=False):
        a1_text = clean_text(getattr(row, "A1", ""))
        anchor_text = clean_text(getattr(row, "Q1", "")) + " " + a1_text
        q2_text = getattr(row, "Q2", "")
        for name, terms in BUSINESS_EVENT_GROUPS.items():
            detail[f"a1_event_{name}"].append(group_score(anchor_text, terms))
        for name, terms in Q2_INTENT_GROUPS.items():
            detail[f"q2_intent_{name}"].append(group_score(q2_text, terms))
        detail["structured_followup_action_score"].append(term_score(q2_text, STRUCTURED_FOLLOWUP_ACTION_TERMS, 2.0))
        detail["structured_reference_score"].append(term_score(q2_text, STRUCTURED_REFERENCE_TERMS, 1.0))
        detail["structured_answer_state_score"].append(term_score(a1_text, A1_ANSWER_STATE_TERMS, 2.0))
        detail["structured_answer_commitment_score"].append(term_score(a1_text, A1_COMMITMENT_STATE_TERMS, 2.0))
        detail["structured_answer_numeric_score"].append(1.0 if re.search(r"\d", a1_text) else 0.0)
        detail["structured_q2_continuation_need_score"].append(term_score(q2_text, Q2_CONTINUATION_NEED_TERMS, 2.0))
        object_score, object_terms = shared_object_score(anchor_text, q2_text)
        a1_object_score, a1_object_terms = shared_object_score(a1_text, q2_text)
        detail["structured_shared_object_score"].append(object_score)
        detail["structured_a1_shared_object_score"].append(a1_object_score)
        detail["structured_bridge_rule_score"].append(0.0)
        capacity_signal = max(
            term_score(anchor_text, ["产能增长", "产能", "投产", "达产", "募集资金项目"], 1.0),
            term_score(q2_text, ["成长性", "增长点", "未来成长"], 1.0),
        )
        if not (term_score(anchor_text, ["产能增长", "产能", "投产", "达产", "募集资金项目"], 1.0) > 0 and term_score(q2_text, ["成长性", "增长点", "未来成长"], 1.0) > 0):
            capacity_signal = 0.0
        detail["structured_capacity_growth_signal"].append(capacity_signal)
        shared_terms.append(object_terms)
        a1_shared_terms.append(a1_object_terms)

    arrays = {name: np.asarray(values, dtype=float) for name, values in detail.items()}
    distance = pd.to_numeric(df["distance"], errors="coerce").fillna(99).to_numpy(dtype=float)
    for event_name, intent_name, rule_score in BUSINESS_BRIDGE_RULES:
        event = arrays[f"a1_event_{event_name}"]
        intent = arrays[f"q2_intent_{intent_name}"]
        matched = (event >= 0.5) & (intent >= 0.5) & near_context & (semantic >= min_semantic)
        arrays["structured_bridge_rule_score"][matched] = np.maximum(arrays["structured_bridge_rule_score"][matched], rule_score)

    action = arrays["structured_followup_action_score"]
    reference = arrays["structured_reference_score"]
    shared = arrays["structured_shared_object_score"]
    a1_shared = arrays["structured_a1_shared_object_score"]
    rule = arrays["structured_bridge_rule_score"]
    capacity_growth_close = (
        (arrays["a1_event_project_capacity"] >= 0.5)
        & (arrays["q2_intent_growth_profit_question"] >= 0.5)
        & (arrays["structured_capacity_growth_signal"] >= 1.0)
        & (action >= 0.45)
        & (distance <= 1)
    )
    strong_reference = (reference >= 0.50) & ((shared >= 0.24) | (a1_shared >= 0.12))
    object_with_business_logic = (
        (shared >= 0.50)
        & (rule > 0)
        & ((a1_shared >= 0.24) | (reference >= 0.50) | (semantic >= 0.03))
    )
    strong_rule_with_weak_object = (
        (rule >= 0.30)
        & ((shared >= 0.12) | (a1_shared >= 0.12))
        & ((action >= 0.45) | (reference >= 0.50))
        & (distance <= 3)
    )
    medium_rule_with_answer_object = (
        (rule >= 0.24)
        & (a1_shared >= 0.12)
        & ((action >= 0.45) | (reference >= 0.50))
        & (distance <= 2)
    )
    concrete_dependency = (
        object_with_business_logic
        | strong_rule_with_weak_object
        | medium_rule_with_answer_object
        | strong_reference
        | capacity_growth_close
    )
    followup_action = (action >= 0.45) | (reference >= 0.50)
    eligible = (rule > 0) & concrete_dependency & followup_action
    bridge_scores[eligible] = np.maximum(
        rule[eligible] + 0.05 * shared[eligible] + 0.03 * action[eligible] + 0.02 * reference[eligible],
        semantic[eligible],
    )
    bridge_scores = np.clip(bridge_scores * score_scale, 0.0, 0.45)

    arrays["structured_bridge_rule_score"] = rule
    arrays["structured_shared_terms"] = np.asarray(shared_terms, dtype=object)
    arrays["structured_a1_shared_terms"] = np.asarray(a1_shared_terms, dtype=object)
    arrays["structured_bridge_eligible"] = eligible.astype(float)
    return bridge_scores, arrays


def question_mark_score(text: object) -> float:
    value = clean_text(text)
    if not value:
        return 0.0
    marks = len(re.findall(r"[?？]", value))
    return float(min(marks / 2.0, 1.0))


def compute_interaction_channels(
    df: pd.DataFrame,
    semantic_prob: np.ndarray,
    structured_bridge_scale: float = 1.0,
    structured_min_semantic: float = 0.0,
) -> pd.DataFrame:
    """Add explicit interaction channels for complaint and challenge follow-ups."""
    out = df.copy()
    semantic = np.asarray(semantic_prob, dtype=float)
    distance = pd.to_numeric(out["distance"], errors="coerce").fillna(99).to_numpy(dtype=float)
    near_context = distance <= 5

    a1_unsat = []
    q2_pressure = []
    a1_claim = []
    q2_challenge = []
    q2_explain_only = []
    evidence = []
    for row in out.itertuples(index=False):
        a1 = getattr(row, "A1", "")
        q2 = getattr(row, "Q2", "")
        a1_len = len(clean_text(a1))
        unsat = term_score(a1, ANSWER_UNSATISFIED_TERMS, 2.0)
        if a1_len <= 18 and unsat > 0:
            unsat = max(unsat, 0.75)
        elif a1_len <= 35 and unsat > 0:
            unsat = max(unsat, 0.60)
        pressure = max(term_score(q2, Q2_PRESSURE_TERMS, 2.0), question_mark_score(q2) * 0.25)
        explain_only = term_score(q2, Q2_EXPLAIN_ONLY_TERMS, 2.0)
        claim = term_score(a1, A1_CLAIM_TERMS, 2.0)
        challenge = max(term_score(q2, Q2_CHALLENGE_TERMS, 2.0), question_mark_score(q2) * 0.35)
        ev = max(term_score(q2, EVIDENCE_TERMS, 2.0), term_score(a1, EVIDENCE_TERMS, 3.0))
        a1_unsat.append(unsat)
        q2_pressure.append(pressure)
        a1_claim.append(claim)
        q2_challenge.append(challenge)
        q2_explain_only.append(explain_only)
        evidence.append(ev)

    out["answer_unsatisfied_score"] = np.asarray(a1_unsat, dtype=float)
    out["q2_pressure_score"] = np.asarray(q2_pressure, dtype=float)
    out["a1_claim_score"] = np.asarray(a1_claim, dtype=float)
    out["q2_challenge_score"] = np.asarray(q2_challenge, dtype=float)
    out["q2_explain_only_score"] = np.asarray(q2_explain_only, dtype=float)
    out["evidence_score"] = np.asarray(evidence, dtype=float)

    complaint_prob = np.zeros(len(out), dtype=float)
    unsat_score = out["answer_unsatisfied_score"].to_numpy(dtype=float)
    pressure_score = out["q2_pressure_score"].to_numpy(dtype=float)
    explain_only_score = out["q2_explain_only_score"].to_numpy(dtype=float)
    pressure_without_explain = np.clip(pressure_score - 0.70 * explain_only_score, 0.0, 1.0)
    pressure_complaint = (
        (unsat_score >= 0.45)
        & (pressure_score >= 0.75)
        & near_context
    )
    strong_complaint = (
        (unsat_score >= 0.58)
        & (pressure_score >= 0.55)
        & near_context
    )
    standalone_pressure = (
        (pressure_without_explain >= 0.92)
        & (semantic >= 0.01)
        & near_context
    )
    explicit_pressure = (
        (pressure_without_explain >= 0.95)
        & (distance <= 2)
    )
    explain_followup_pressure = (
        (explain_only_score >= 0.65)
        & (unsat_score >= 0.45)
        & near_context
    )

    complaint_prob[pressure_complaint] = np.maximum(complaint_prob[pressure_complaint], 0.58)
    complaint_prob[strong_complaint] = np.maximum(complaint_prob[strong_complaint], 0.80)
    complaint_prob[standalone_pressure] = np.maximum(complaint_prob[standalone_pressure], 0.56)
    complaint_prob[explicit_pressure] = np.maximum(complaint_prob[explicit_pressure], 0.55)
    complaint_prob[explain_followup_pressure] = np.maximum(complaint_prob[explain_followup_pressure], 0.53)

    challenge_prob = np.zeros(len(out), dtype=float)
    challenge_score = out["q2_challenge_score"].to_numpy(dtype=float)
    claim_score = out["a1_claim_score"].to_numpy(dtype=float)
    evidence_score = out["evidence_score"].to_numpy(dtype=float)
    challenge_context_support = semantic >= 0.02
    strong_challenge = (
        (challenge_score >= 0.55)
        & ((claim_score >= 0.35) | (evidence_score >= 0.35))
        & challenge_context_support
        & near_context
    )
    evidence_challenge = (
        (evidence_score >= 0.55)
        & (challenge_score >= 0.35)
        & (semantic >= 0.02)
        & near_context
    )
    semantic_challenge = (
        (challenge_score >= 0.45)
        & (semantic >= 0.06)
        & near_context
    )
    challenge_prob[semantic_challenge] = np.maximum(challenge_prob[semantic_challenge], 0.58)
    challenge_prob[evidence_challenge] = np.maximum(challenge_prob[evidence_challenge], 0.70)
    challenge_prob[strong_challenge] = np.maximum(challenge_prob[strong_challenge], 0.78)

    structured_prob, structured_details = compute_business_bridge_scores(
        out,
        semantic,
        near_context,
        score_scale=structured_bridge_scale,
        min_semantic=structured_min_semantic,
    )
    for name, values in structured_details.items():
        out[name] = values

    structured_other_support = (semantic >= 0.005) | (complaint_prob >= 0.12) | (challenge_prob >= 0.12)
    structured_active = structured_prob >= 0.12
    structured_standalone = structured_active & ~structured_other_support
    structured_answer_state = np.maximum.reduce([
        out["structured_answer_state_score"].to_numpy(dtype=float),
        out["structured_answer_commitment_score"].to_numpy(dtype=float),
    ])
    structured_need = out["structured_q2_continuation_need_score"].to_numpy(dtype=float)
    structured_rule = out["structured_bridge_rule_score"].to_numpy(dtype=float)
    structured_shared = out["structured_shared_object_score"].to_numpy(dtype=float)
    structured_a1_shared = out["structured_a1_shared_object_score"].to_numpy(dtype=float)
    structured_reference = out["structured_reference_score"].to_numpy(dtype=float)
    structured_capacity = out["structured_capacity_growth_signal"].to_numpy(dtype=float)
    followup_action = np.maximum.reduce([
        out["structured_followup_action_score"].to_numpy(dtype=float),
        structured_reference,
    ])

    answer_state_guard = (
        (structured_rule > 0)
        & (structured_need >= 0.35)
        & (structured_answer_state >= 0.35)
        & ((structured_a1_shared >= 0.12) | (structured_shared >= 0.24) | (structured_reference >= 0.50))
        & near_context
    )
    capacity_growth_guard = (
        (structured_capacity >= 1.0)
        & (out["q2_intent_growth_profit_question"].to_numpy(dtype=float) >= 0.50)
        & (structured_answer_state >= 0.35)
        & (distance <= 1)
    )
    structured_standalone_guard = answer_state_guard | capacity_growth_guard
    structured_prob = structured_prob.copy()
    structured_prob[structured_standalone & ~structured_standalone_guard] = 0.0

    semantic = semantic.copy()
    complaint_prob = complaint_prob.copy()
    challenge_prob = challenge_prob.copy()
    structured_prob = structured_prob.copy()

    out["structured_answer_state_guard"] = answer_state_guard.astype(float)
    out["structured_capacity_growth_guard"] = capacity_growth_guard.astype(float)
    out["structured_standalone_guard"] = structured_standalone_guard.astype(float)
    out["structured_standalone_blocked"] = (structured_standalone & ~structured_standalone_guard).astype(float)

    out["stage1_semantic_prob"] = semantic
    out["stage1_complaint_prob"] = complaint_prob
    out["stage1_challenge_prob"] = challenge_prob
    out["stage1_structured_prob"] = structured_prob

    # The reported Stage 1 configuration uses the supervised semantic edge
    # probability directly as the retention score.
    out["stage1_prob"] = stage1_selection_score(out)
    return out


def fit_stage1(X_train: np.ndarray, y_train: np.ndarray) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    ).fit(X_train, y_train)


def predict_positive_proba(model: object, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    return proba[:, 1].astype(float)


def stage1_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def stage1_selection_score(df: pd.DataFrame) -> np.ndarray:
    """Return the default Stage 1 retention score.

    In the paper configuration, the retention score is exactly the supervised
    edge probability stored in ``stage1_semantic_prob``. Complaint, challenge,
    and structured channels are exported as edge-local auxiliary signals but
    do not alter the default Stage 1 retention score.
    """
    return np.clip(stage1_numeric_series(df, "stage1_semantic_prob", 0.0).to_numpy(dtype=float), 0.0, 1.0)



def add_source_rank(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Return deterministic later-query ranks for each source anchor.

    Ties follow the existing row order, which is fixed by the input candidate
    table. The rank implements the fixed Top-K later-query cap for each earlier
    Q&A source, consistent with Stage 1 source-conditioned retention.
    """
    if "Q1_row" not in df.columns:
        raise ValueError("Stage 1 data must contain Q1_row for source-wise Top-K retention.")
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    return (
        scores.groupby(
            [df["session_id"].astype(str), df["Q1_row"].astype(str)],
            sort=False,
        )
        .rank(method="first", ascending=False)
        .astype(int)
    )


def apply_fixed_retention_policy(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    top_k: int,
) -> np.ndarray:
    """Apply the revised Stage 1 threshold + Top-K retention rule.

    An edge is retained iff:
      (1) its learned Stage 1 score is at least ``threshold``; and
      (2) it is among the top ``top_k`` candidates for the same anchor.

    There is no adaptive confidence formula, protected-candidate rule,
    entropy weighting, or fallback edge insertion.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    rank = add_source_rank(df, score_col)
    return ((score >= float(threshold)) & (rank <= int(top_k))).to_numpy(dtype=bool)


def tune_retention_policy(
    validation_df: pd.DataFrame,
    score_col: str,
    target_recall: float,
    max_top_k: int,
) -> Tuple[float, int, pd.DataFrame]:
    """Select Stage 1 retention parameters on Validation only.

    For each K in 1,...,max_top_k, the function finds the *largest* score
    threshold that can retain at least ``target_recall`` of all positive
    validation edges after the Top-K constraint. Among feasible K values, the
    selected configuration minimizes the total number of retained validation
    candidates. Ties are resolved by smaller K and then by the larger threshold.

    This yields an exact validation-side constrained selection rule without a
    manually specified threshold grid.
    """
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("target_recall must be in (0, 1].")
    if int(max_top_k) < 1:
        raise ValueError("max_top_k must be >= 1.")

    work = validation_df.copy().reset_index(drop=True)
    labels = pd.to_numeric(work["edge_label"], errors="coerce").fillna(0).astype(int)
    scores = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0).astype(float)
    ranks = add_source_rank(work, score_col)

    total_positive = int(labels.sum())
    if total_positive <= 0:
        raise RuntimeError("Validation split contains no positive source-dependency edges.")

    required_positive = int(math.ceil(float(target_recall) * total_positive - 1e-12))
    rows: List[Dict[str, object]] = []

    for top_k in range(1, int(max_top_k) + 1):
        topk_mask = ranks <= top_k
        eligible_positive_scores = scores[(labels == 1) & topk_mask].sort_values(ascending=False)
        max_positive_recovered = int(len(eligible_positive_scores))
        max_recall = float(max_positive_recovered / total_positive)

        feasible = max_positive_recovered >= required_positive
        if feasible:
            # Largest threshold that still retains at least the required number
            # of positive edges. Since the decision uses >=, tied scores are
            # retained together.
            threshold = float(eligible_positive_scores.iloc[required_positive - 1])
            pred = topk_mask & (scores >= threshold)
        else:
            threshold = float("nan")
            # Diagnostic maximum achievable recall at this K.
            pred = topk_mask

        metrics = binary_metrics(
            labels.to_numpy(dtype=int),
            pred.to_numpy(dtype=int),
            scores.to_numpy(dtype=float),
        )
        rows.append(
            {
                "TopK": int(top_k),
                "Threshold": threshold,
                "Feasible": bool(feasible),
                "TargetRecall": float(target_recall),
                "RequiredPositiveEdges": int(required_positive),
                "ValidationPositiveEdges": int(total_positive),
                "MaxRecallAtK": float(max_recall),
                "Recall": float(metrics["recall"]),
                "Precision": float(metrics["precision"]),
                "F1": float(metrics["f1"]),
                "RetainedEdges": int(metrics["pred_edges"]),
                "RetentionRatio": float(metrics["pred_edges"] / len(work)) if len(work) else 0.0,
                "TruePositives": int(metrics["tp"]),
                "FalsePositives": int(metrics["fp"]),
                "FalseNegatives": int(metrics["fn"]),
            }
        )

    search = pd.DataFrame(rows)
    feasible_search = search[search["Feasible"]].copy()
    if feasible_search.empty:
        raise RuntimeError(
            "No Top-K value can satisfy the requested validation recall. "
            "Increase --max-top-k or lower --target-recall."
        )

    feasible_search = feasible_search.sort_values(
        ["RetainedEdges", "TopK", "Threshold"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    best = feasible_search.iloc[0]
    return float(best["Threshold"]), int(best["TopK"]), search


def save_stage(df: pd.DataFrame, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final FCN Stage 1: fixed 176/44/55 Train/Validation/Test split, "
            "Train-only learned scorer, Validation-only 0.98-recall retention "
            "selection, and no post-validation refit."
        )
    )
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--session-manifest", required=True)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--stage1-outdir", default=str(DEFAULT_STAGE1_OUTDIR))
    parser.add_argument("--encode-batch-size", type=int, default=64)

    # These values generate auxiliary edge-local channels consumed by Stage 2A.
    # They do not determine Stage 1 candidate retention.
    parser.add_argument("--structured-bridge-scale", type=float, default=0.5)
    parser.add_argument("--structured-min-semantic", type=float, default=0.0)

    return parser.parse_args()

def build_fixed_three_way_split(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact manifest already used for encoder fine-tuning."""
    if "session_id" not in df.columns:
        raise ValueError("Candidate data must contain session_id.")
    session_ids = df["session_id"].astype(str).map(clean_text)
    manifest_ids = set(manifest["session_id"])
    data_ids = set(session_ids)
    unknown = sorted(data_ids - manifest_ids)
    absent = sorted(manifest_ids - data_ids)
    if unknown:
        raise ValueError(f"Candidate data contain sessions absent from manifest: {unknown[:10]}")
    if absent:
        raise ValueError(f"Manifest contains sessions absent from candidate data: {absent[:10]}")
    split_by_session = manifest.set_index("session_id")["split"]
    row_splits = session_ids.map(split_by_session)
    if row_splits.isna().any():
        raise RuntimeError("At least one candidate row could not be assigned to a split.")
    train_idx = np.flatnonzero(row_splits.eq("train").to_numpy())
    validation_idx = np.flatnonzero(row_splits.eq("validation").to_numpy())
    test_idx = np.flatnonzero(row_splits.eq("test").to_numpy())
    return train_idx, validation_idx, test_idx


def save_split_manifest(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Save the frozen session assignment for reproducibility."""
    rows: List[Dict[str, str]] = []
    for split_name, part in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        for session_id in part["session_id"].astype(str).drop_duplicates():
            rows.append({"session_id": session_id, "split": split_name})

    pd.DataFrame(rows).to_csv(
        outdir / "stage1_split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_stage1(args: argparse.Namespace) -> Path:
    data_path = Path(args.data_file).expanduser().resolve()
    session_manifest_path = Path(args.session_manifest).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Candidate-data file does not exist: {data_path}")
    manifest = load_session_manifest(session_manifest_path)
    model_path, encoder_provenance = verify_encoder_provenance(
        Path(args.model_dir),
        data_path,
        session_manifest_path,
    )
    outdir = Path(args.stage1_outdir)
    cache_dir = outdir / "embedding_cache"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, dtype=str, keep_default_na=False)

    df = add_labels(df)
    df = add_text_features(df)
    df = df.reset_index(drop=True)

    # Fixed encoder representations. No labels are used during encoding.
    anchor_emb, cand_emb = encode_embeddings(
        df=df,
        model_dir=str(model_path),
        cache_dir=cache_dir,
        batch_size=args.encode_batch_size,
        data_sha256=sha256_file(data_path),
        encoder_provenance_sha256=sha256_file(
            model_path / "encoder_provenance_manifest.json"
        ),
    )
    X = build_features(df, anchor_emb, cand_emb)

    # Frozen 176 / 44 / 55 Train / Validation / Test split.
    train_idx, validation_idx, test_idx = build_fixed_three_way_split(df, manifest)

    train_base = df.iloc[train_idx].copy().reset_index(drop=True)
    validation_base = df.iloc[validation_idx].copy().reset_index(drop=True)
    test_base = df.iloc[test_idx].copy().reset_index(drop=True)

    X_train = X[train_idx]
    X_validation = X[validation_idx]
    X_test = X[test_idx]

    y_train = train_base["edge_label"].to_numpy(dtype=int)
    y_validation = validation_base["edge_label"].to_numpy(dtype=int)
    y_test = test_base["edge_label"].to_numpy(dtype=int)

    print_block(
        "Frozen Train / Validation / Test Split",
        {
            "train_rows": len(train_base),
            "validation_rows": len(validation_base),
            "test_rows": len(test_base),
            "train_true_edges": int(y_train.sum()),
            "validation_true_edges": int(y_validation.sum()),
            "test_true_edges": int(y_test.sum()),
            "train_sessions": train_base["session_id"].nunique(),
            "validation_sessions": validation_base["session_id"].nunique(),
            "test_sessions": test_base["session_id"].nunique(),
        },
    )

    save_split_manifest(train_base, validation_base, test_base, outdir)

    # Fit the learned Stage 1 scorer ONCE on Train only.
    model = fit_stage1(X_train, y_train)

    train_prob = predict_positive_proba(model, X_train)
    validation_prob = predict_positive_proba(model, X_validation)

    # Select the operating point on Validation only.
    validation_for_selection = validation_base.copy()
    validation_for_selection["stage1_prob"] = np.clip(
        validation_prob,
        0.0,
        1.0,
    )

    selected_threshold, selected_top_k, validation_search = tune_retention_policy(
        validation_for_selection,
        score_col="stage1_prob",
        target_recall=TARGET_VALIDATION_RECALL,
        max_top_k=MAX_TOP_K,
    )

    validation_search.to_csv(
        outdir / "stage1_validation_retention_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_row = validation_search[
        (validation_search["TopK"] == selected_top_k)
        & np.isclose(
            validation_search["Threshold"].astype(float),
            selected_threshold,
            rtol=0.0,
            atol=1e-15,
        )
    ].iloc[0]

    print_block(
        "Validation-selected Final Stage 1 Retention",
        {
            "target_recall": TARGET_VALIDATION_RECALL,
            "selected_threshold": float(selected_threshold),
            "selected_top_k": int(selected_top_k),
            "validation_recall": float(selected_row["Recall"]),
            "validation_precision": float(selected_row["Precision"]),
            "validation_f1": float(selected_row["F1"]),
            "validation_retained_edges": int(selected_row["RetainedEdges"]),
            "validation_retention_ratio": float(selected_row["RetentionRatio"]),
        },
    )

    # Freeze scorer + retention parameters. NO REFIT occurs after validation.
    # Test is scored only after the operating point has been frozen.
    test_prob = predict_positive_proba(model, X_test)

    train_scored = compute_interaction_channels(
        train_base,
        train_prob,
        structured_bridge_scale=args.structured_bridge_scale,
        structured_min_semantic=args.structured_min_semantic,
    )
    validation_scored = compute_interaction_channels(
        validation_base,
        validation_prob,
        structured_bridge_scale=args.structured_bridge_scale,
        structured_min_semantic=args.structured_min_semantic,
    )
    test_scored = compute_interaction_channels(
        test_base,
        test_prob,
        structured_bridge_scale=args.structured_bridge_scale,
        structured_min_semantic=args.structured_min_semantic,
    )

    for part in (train_scored, validation_scored, test_scored):
        part["stage1_prob"] = stage1_selection_score(part)
        part["stage1_pred_edge"] = apply_fixed_retention_policy(
            part,
            score_col="stage1_prob",
            threshold=selected_threshold,
            top_k=selected_top_k,
        ).astype(int)

    train_metrics = binary_metrics(
        train_scored["edge_label"],
        train_scored["stage1_pred_edge"],
        train_scored["stage1_prob"],
    )
    validation_metrics = binary_metrics(
        validation_scored["edge_label"],
        validation_scored["stage1_pred_edge"],
        validation_scored["stage1_prob"],
    )
    test_metrics = binary_metrics(
        test_scored["edge_label"],
        test_scored["stage1_pred_edge"],
        test_scored["stage1_prob"],
    )

    print_block("Stage 1 - Train Candidate Retention", train_metrics)
    print_block("Stage 1 - Validation Candidate Retention", validation_metrics)
    print_block("Stage 1 - Formal Test Candidate Retention", test_metrics)

    # Exactly three scored partitions for downstream Stage 2A.
    save_stage(train_scored, outdir, "stage1_train_scored")
    save_stage(validation_scored, outdir, "stage1_validation_scored")
    save_stage(test_scored, outdir, "stage1_test_scored")

    selected_params = {
        "protocol": "manifest_fixed_176_44_55_train_validation_test_no_refit",
        "retention_orientation": "source_conditioned",
        "retention_group_keys": ["session_id", "Q1_row"],
        "input_format": INPUT_FORMAT,
        "text_protocol_sha256": TEXT_PROTOCOL_SHA256,
        "seed": SEED,
        "candidate_data_sha256": sha256_file(data_path),
        "session_manifest_sha256": sha256_file(session_manifest_path),
        "encoder_training_objective": encoder_provenance["training_objective"],
        "encoder_train_only": encoder_provenance["train_only"],
        "rationale_derived_supervision": encoder_provenance[
            "rationale_derived_supervision"
        ],
        "session_counts": {
            "train": EXPECTED_TRAIN_SESSIONS,
            "validation": EXPECTED_VALIDATION_SESSIONS,
            "test": EXPECTED_TEST_SESSIONS,
        },
        "target_validation_recall": TARGET_VALIDATION_RECALL,
        "max_top_k": MAX_TOP_K,
        "selected_threshold": float(selected_threshold),
        "selected_top_k": int(selected_top_k),
        "scorer_refit_after_validation": False,
        "test_labels_used_for_fitting_or_selection": False,
        "validation_metrics": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in validation_metrics.items()
        },
        "test_metrics": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in test_metrics.items()
        },
    }

    (outdir / "stage1_final_selected_params.json").write_text(
        json.dumps(selected_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 86)
    print("Final FCN Stage 1 completed successfully")
    print("=" * 86)
    print(f"output_dir: {outdir}")
    print("Scorer fitted once on Train only; no post-validation refit.")
    print(
        "Stage 2A inputs: stage1_train_scored.csv, "
        "stage1_validation_scored.csv, stage1_test_scored.csv"
    )

    return outdir


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("[1/1] Run final FCN Stage 1...")
    stage1_outdir = run_stage1(args)
    print(f"Final Stage 1 completed successfully: {stage1_outdir}")


if __name__ == "__main__":
    main()
