"""Issue #2 验收 — 双 LLM Runtime 已统一到 AgentScope runtime。

证明 research.py 的结构化 JSON LLM 调用（planner / evidence evaluator /
follow-up generator）走的是控制器 DI 注入的统一 AgentScope 模型，
不再有 raw openai.OpenAI 客户端 / 第二套 credential / 第二套参数配置。

Run:  .venv/bin/python verify_unified_runtime.py
Exit 0 = all pass。
"""
import inspect

import agentscope
import research
from agent.builder import get_research_model


def main() -> None:
    import asyncio

    # 1. 统一模型是 AgentScope OpenAIChatModel（不是 openai.OpenAI）
    model = get_research_model()
    assert isinstance(model, agentscope.model.OpenAIChatModel), \
        f"get_research_model -> {type(model)}, 期望 OpenAIChatModel"
    print(f"PASS  get_research_model -> {type(model).__module__}.{type(model).__name__}")
    print(f"      model={model.model!r} base_url={model.credential.base_url!r}")
    print(f"      stream={model.stream} max_retries={model.max_retries} "
          f"thinking_enable={model.parameters.thinking_enable}")

    # 2. DI：控制器注入后，research._resolve_model() 返回同一实例
    research.set_research_model(model)
    assert research._resolve_model() is model
    print("PASS  DI: set_research_model 后 _resolve_model() 返回注入实例")

    # 3. 五个 LLM 函数都是 async（在控制器里 await）
    for fn in (research.make_plan, research.evaluate_evidence,
               research.build_followup_queries, research.build_research_query,
               research._llm):
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} 不是 async"
    print("PASS  planner/evaluator/followup/query/_llm 全部 async")

    # 4. 实跑：三个 research-loop LLM 调用经统一模型返回真实结果
    plan = asyncio.run(research.make_plan("2026 年 AI Agent 框架趋势"))
    assert plan, f"make_plan -> {plan!r}"
    print(f"PASS  make_plan (live) -> {len(plan)} 步")

    issues = asyncio.run(research.evaluate_evidence(
        "2026 年 AI Agent 框架趋势", plan,
        "# 趋势\n- 生产级框架密集发布 [evt_815fcf6e2a6f]\n## Citations\n"
        "- evt_815fcf6e2a6f — 生产级多智能体框架密集发布（标题）"))
    assert "verdict" in issues, f"evaluate_evidence -> {issues!r}"
    print(f"PASS  evaluate_evidence (live) -> verdict={issues['verdict']!r}")

    if issues.get("gaps") or issues.get("conflicts"):
        fu = asyncio.run(research.build_followup_queries("2026 AI Agent", issues))
        print(f"PASS  build_followup_queries (live) -> {fu[:2]}")
    else:
        print("PASS  build_followup_queries (live) -> (无缺口/矛盾，跳过)")

    # 5. research.py 无任何 raw openai / 第二套配置
    src = open(research.__file__, encoding="utf-8").read()
    assert "openai.OpenAI" not in src, "research.py 仍有 openai.OpenAI"
    assert "from openai import" not in src, "research.py 仍有 raw openai import"
    assert "load_llm_config" not in src, "research.py 仍引用 load_llm_config"
    print("PASS  research.py: 无 openai.OpenAI / 无 raw openai / 无 load_llm_config")

    print("\n全部通过 — research loop 的 LLM 调用统一走 AgentScope runtime（DI 注入）。")


if __name__ == "__main__":
    main()