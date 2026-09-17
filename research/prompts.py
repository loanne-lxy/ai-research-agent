"""Prompt templates + static fallback for the research pipeline.

Each LLM stage reads its template here. No model / loop logic lives in this
file — just the strings, and the generic DEFAULT_PLAN fallback used when a
model call or its JSON parse fails.
"""
from __future__ import annotations

# ponytail: static generic fallback — used when the model call or JSON
# parse fails. If a domain-specific default measurably plans better,
# switch on corpus_stats categories.
DEFAULT_PLAN = [
    "收集与研究问题直接相关的事件与文章",
    "按方向/主题归类",
    "评估各方向的证据强度",
    "识别证据缺口",
    "针对缺口补充检索",
    "形成带引用的综合结论",
]

PLAN_PROMPT = """为以下研究问题制定一份研究计划。只返回一个 JSON 数组（不要其他文字），
数组元素是 3~7 个简短中文步骤（字符串），按执行顺序排列，覆盖：
收集证据 → 归类 → 评估证据 → 识别缺口 → 补充检索 → 综合结论。
步骤要针对该问题具体化，不要空话。

问题：{question}"""

EVAL_PROMPT = """你是证据审核员。下面是研究回答的证据地图（章节标题+全部正文行；不带引用ID的正文行即未引用陈述）。只返回一个JSON对象（不要其他文字），note不超过15字：
{{"directions":[{{"name":"方向","events":数,"articles":数,"sufficient":true或false,"conflict":true或false,"note":"简"}}],"gaps":["需补充检索的缺口"],"conflicts":["相互矛盾的点"],"verdict":"sufficient或needs_work"}}
标准：某方向证据行<3 → sufficient=false；单来源（独立source仅1个）支撑的结论 → sufficient=false 且列入 gaps（写"单来源: <结论>，需补充独立来源"）；证据互相矛盾 → conflict=true；未带引用ID的事实性陈述 → 列入 gaps（写"未引用: <该事实>"，寒暄/方法论解释不算）；存在 sufficient=false / conflict=true / gaps → verdict=needs_work。只数地图里实际出现的引用，草稿已声明知识库未覆盖的缺口要列入 gaps。

研究问题：{question}
研究计划：
{plan}

结论-独立来源数：
{claims}

证据地图：
{draft}"""

FOLLOWUP_PROMPT = """下面是研究回答被评估后发现的证据缺口与矛盾。为每个缺口/矛盾生成一条检索关键词（15字以内，直接给检索词，可中英混合，可含年份/季度）；矛盾的检索词优先用于裁决哪方属实。只返回一个JSON对象：
{"follow_ups":["检索词1","检索词2"]}
不要解释。

研究问题：{question}
缺口/矛盾：
{items}"""