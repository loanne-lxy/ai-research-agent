"""Self-check for the ReMe long-term memory integration (run: python /tmp/check_reme.py).

Verifies wiring without hitting the network:
  1. agent.memory.get_memory_middleware is a singleton with the right config
  2. its list_tools() exposes exactly memory_search (mode=both)
  3. build_agent() (now async) attaches the middleware + the memory tool
  4. _make_revision_agent passes the SAME middleware instance (write-back on revisions)
  5. web_service.build_app() builds and the factory returns that same instance
"""
import asyncio
import sys

sys.path.insert(0, "/home/loanne/ai-research-agent")


async def main() -> None:
    # 1. singleton + config
    from agent.memory import get_memory_middleware, MEMORY_DIR
    mw = get_memory_middleware()
    assert get_memory_middleware() is mw, "not a singleton"
    assert str(MEMORY_DIR) == mw._workspace_dir, "workspace_dir mismatch"
    assert mw._parameters.mode == "both", "mode should be both"
    assert mw._parameters.chat_model is not None, "chat_model not injected"
    assert not mw._parameters.chat_model.parameters.thinking_enable, "thinking should be off for extraction"

    # 2. tools
    tools = await mw.list_tools()
    names = [t.name for t in tools]
    assert names == ["memory_search"], f"unexpected tools: {names}"

    # 3. CLI agent
    from agent.builder import build_agent
    agent = await build_agent()
    schema_names = [s["function"]["name"] for s in await agent.toolkit.get_tool_schemas()]
    assert "memory_search" in schema_names, f"memory_search missing: {schema_names}"
    assert len(agent._reply_middlewares) == 1, "middleware not in reply chain"
    assert agent._reply_middlewares[0] is mw, "not the same middleware instance"

    # 4. revision agent reuses the same instance
    from agent.research_agent import ResearchAgent
    assert isinstance(agent, ResearchAgent) or True  # CLI builds base Agent
    # build a ResearchAgent the way the service does (middlewares arg) and check revision
    from agentscope.model import OpenAIChatModel
    from agentscope.credential import OpenAICredential
    from agentscope.permission import PermissionContext, PermissionMode
    from agentscope.state import AgentState
    import config as C
    llm = C.load_llm_config()
    cred = OpenAICredential(api_key=llm.api_key, base_url=llm.base_url)
    model = OpenAIChatModel(credential=cred, model=llm.model)
    ra = ResearchAgent(
        name="ResearchExpert",
        system_prompt="x",
        model=model,
        toolkit=agent.toolkit,
        middlewares=[mw],
        state=AgentState(permission_context=PermissionContext(mode=PermissionMode.DONT_ASK)),
    )
    rev = await ra._make_revision_agent()
    assert rev._reply_middlewares == [mw], "revision agent lost the middleware"
    assert rev.state.session_id == ra.state.session_id, "revision lost session id"

    # 5. web service wiring
    from web_service import build_app, _extra_agent_middlewares
    app, storage, workspace = build_app()
    got = await _extra_agent_middlewares("loanne", "aid", "sid")
    assert got == [mw], "web factory must return the same singleton"
    assert app.state.extra_agent_middlewares is _extra_agent_middlewares, "factory not on app state"

    print("self-check ok: singleton + memory_search tool + CLI/web wiring coherent")


if __name__ == "__main__":
    asyncio.run(main())