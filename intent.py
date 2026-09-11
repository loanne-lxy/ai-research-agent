"""Intent classification — the v2 entry fork.

Decides whether an incoming question is a fact / simple QA (route to the
standard v1 path) or an open research question (route to the research
path). One cheap LLM call per question; on any failure it falls back to
"fact" so a classifier outage never blocks the proven v1 path.

Run:  .venv/bin/python intent.py "2026 Q2 以来 Agent 领域的发展趋势是什么？"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from config import load_llm_config

FACT = "fact"
RESEARCH = "research"

_PROMPT = """判断用户问题的类型，只返回一个 JSON 对象，不要任何其他文字：
{"intent": "fact 或 research", "reason": "一句话理由（中文）"}
判定标准：
- fact：对知识库既有内容的直接查询，一两次检索即可回答（如"本周有哪些多模态事件？""某篇论文的核心方法是什么？"）。
- research：开放性问题，需要多轮检索、跨材料对比与综合才能回答（如"2026 Q2 以来 Agent 领域的发展趋势是什么？""对比 A 与 B 两条技术路线的优劣"）。

问题：{question}"""


@dataclass(frozen=True)
class Intent:
    intent: str  # FACT or RESEARCH
    reason: str


def _parse(text: str) -> Intent:
    """Extract the JSON object from a model reply. Anything unparseable
    or unknown falls back to FACT (the proven, cheaper path)."""
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0)) if m else {}
    raw = str(data.get("intent", FACT)).strip().lower()
    intent = RESEARCH if raw == RESEARCH else FACT
    return Intent(intent, str(data.get("reason", "")).strip() or "—")


def classify_intent(question: str) -> Intent:
    """Classify a question. Never raises — errors fall back to FACT.

    # ponytail: context-blind — a bare follow-up ("刚才那个展开讲") is
    # classified on its own text. Add recent turns as context when that
    # measurably misroutes.
    """
    try:
        from openai import OpenAI
        llm = load_llm_config()
        client = OpenAI(base_url=llm.base_url, api_key=llm.api_key, timeout=60)
        out = client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user",
                       "content": _PROMPT.replace("{question}", question)}],
            temperature=0,
            max_tokens=500,
            # Qwen3.8 是推理模型，thinking token 计入预算；分类只需一个
            # 小 JSON，思考纯属浪费预算（见 research._llm 的说明）
            extra_body={"enable_thinking": False},
        )
        return _parse(out.choices[0].message.content or "")
    except Exception as e:  # noqa: BLE001 - classifier must not break the REPL
        return Intent(FACT, f"分类失败（{type(e).__name__}），按简单问答处理")


if __name__ == "__main__":
    # self-check: parse + fallback logic only, no LLM call
    assert _parse('{"intent": "research", "reason": "开放趋势问题"}').intent == RESEARCH
    assert _parse('结果是：\n```json\n{"intent": "fact", "reason": "直接查询"}\n```').intent == FACT
    assert _parse("garbage, no json at all").intent == FACT
    assert _parse('{"intent": "weird", "reason": "x"}').intent == FACT
    assert _parse('{"intent": "RESEARCH", "reason": "大写也应识别"}').intent == RESEARCH
    print("self-check ok: parse + fallback")
    if len(sys.argv) > 1:  # optional live probe against the real model
        print(classify_intent(sys.argv[1]))