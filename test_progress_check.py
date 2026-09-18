"""Self-check: run_research_loop pushes sidecar snapshots in order
(plan first, per-evaluation, final with citations) via on_progress,
and the follow-up guard in ResearchAgent routes around the loop."""
import asyncio, sys, types
sys.path.insert(0, "/home/loanne/ai-research-agent")

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
import research.controller as ctl

snaps = []
async def fake_reply_stream(self, inputs=None, structured_schema=None, yield_final_msg=False):
    yield Msg(name="a", role="assistant", content=[TextBlock(text="answer [evt_x]")])
Agent.reply_stream = fake_reply_stream

def fake_cls(q): return types.SimpleNamespace(intent="research")
async def fake_brq(q):
    return ("[研究任务] 研究计划（你的工作状态，按此推进，不要向用户重复计划全文）：\n"
            "1. step-a\n2. step-b\n\n研究问题：" + q)
async def fake_eval(*a):
    return {"verdict": "sufficient",
            "directions": [{"name": "d1", "events": 2, "articles": 1,
                            "sufficient": True, "conflict": False, "note": ""}],
            "gaps": [], "conflicts": []}

ctl.classify_intent = fake_cls
ctl.build_research_query = fake_brq
ctl.evaluate_evidence = fake_eval
ctl.validate = lambda a: {"ok": [], "total": 0, "missing": []}
ctl.extract_claims = lambda a: []

# fake agent: needs only .state.session_id for the controller's sake
agent = types.SimpleNamespace(state=types.SimpleNamespace(session_id="s1", reply_id="r1"))

out = {}
async def main():
    agen = ctl.run_research_loop("q", agent, make_agent=None, out=out,
                                 on_progress=snaps.append)
    async for _ in agen:
        pass
    return out

out = asyncio.run(main())

assert len(snaps) >= 3, f"expected >=3 snapshots, got {len(snaps)}"
assert snaps[0]["research_plan"] == ["step-a", "step-b"], snaps[0]["research_plan"]
assert snaps[0]["verdict"] == "running"
assert snaps[0]["claims"] == []
assert snaps[1]["evidence"][0]["name"] == "d1"
assert snaps[-1]["citations"]["total"] == 0
assert out["result"]["claims"] == [] and out["result"]["evidence"], "final out broken"
print(f"controller OK: {len(snaps)} snapshots "
      f"({[s['verdict'] for s in snaps]})")

# ── follow-up guard: non-empty context must NOT enter the research loop ──
loop_called = {"v": False}
async def boom(*a, **k):
    loop_called["v"] = True
    raise AssertionError("research loop must not run on follow-up turn")
    yield

import agent.research_agent as ra
ra.run_research_loop = boom

class FakeState:
    def __init__(self, ctx): self.context, self.session_id, self.reply_id = ctx, "s2", "r2"

agent2 = ra.ResearchAgent.__new__(ra.ResearchAgent)  # skip pydantic ctor
agent2.state = FakeState(ctx=[Msg(name="u", role="user", content=[TextBlock(text="prior")])])
got = []
async def fake_super(self, inputs=None, structured_schema=None, yield_final_msg=False):
    got.append("base")
    yield Msg(name="a", role="assistant", content=[TextBlock(text="follow-up answer")])
Agent.reply_stream = fake_super

async def run2():
    async for ev in ra.ResearchAgent.reply_stream(agent2,
            Msg(name="u", role="user", content=[TextBlock(text="追问?")])):
        pass
asyncio.run(run2())
assert got == ["base"] and not loop_called["v"], (got, loop_called)
print("follow-up guard OK: routed to base ReAct, loop untouched")