"""Verify the JWT auth edge + multi-user bootstrap. Run: .venv/bin/python verify_auth.py"""
import json
import urllib.request
import urllib.error

B = "http://127.0.0.1:8000"
NO_AUTH = object()  # sentinel meaning "send no Authorization header"


def req(method, path, body=None, auth=NO_AUTH):
    hdrs = {"Content-Type": "application/json"}
    if auth is not NO_AUTH:
        hdrs["Authorization"] = "Bearer " + auth
    r = urllib.request.Request(
        B + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers=hdrs)
    try:
        with urllib.request.urlopen(r) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}


def main():
    import auth
    auth.create_user("loanne", "loanne")   # idempotent for the test
    auth.create_user("alice", "wonderland")
    results = []

    def check(name, ok):
        results.append((name, ok))
        print(("PASS" if ok else "FAIL"), name)

    st, l = req("POST", "/api/auth/login", {"username": "loanne", "password": "loanne"})
    T = l.get("token", "")
    check("1 login ok", st == 200 and len(T) > 50)
    check("1b login bad pw -> 401",
          req("POST", "/api/auth/login", {"username": "loanne", "password": "wrong"})[0] == 401)
    check("2 auth/me", req("POST", "/api/auth/me", auth=T)[0] == 200)
    check("3 no auth -> 401", req("POST", "/api/auth/me")[0] == 401)
    check("4 bad auth -> 401", req("POST", "/api/auth/me", auth="garbage.not.jwt")[0] == 401)

    st, b = req("POST", "/api/bootstrap", auth=T)
    check("5 bootstrap loanne", st == 200 and bool(b.get("model", {}).get("model")))

    TA = req("POST", "/api/auth/login", {"username": "alice", "password": "wonderland"})[1].get("token", "")
    stA, bA = req("POST", "/api/bootstrap", auth=TA)
    check("6 bootstrap alice", stA == 200 and bool(bA.get("agent")))
    check("7 alice != loanne agent", bool(bA.get("agent") and b.get("agent") and bA["agent"] != b["agent"]))

    stMine, mine = req("GET", "/sessions/?agent_id=" + b["agent"], auth=T)
    stTheirs, theirs = req("GET", "/sessions/?agent_id=" + b["agent"], auth=TA)
    # Isolation = loanne sees her session (key is `sessions`); alice is
    # blocked (404: her resolve_agent can't see loanne's unshared agent).
    check("8 isolation (loanne=1, alice blocked)",
          stMine == 200 and mine.get("total", 0) == 1
          and (stTheirs == 404 or theirs.get("total", 0) == 0))

    check("9 knowledge stats", req("GET", "/api/knowledge/stats", auth=T)[0] == 200)
    check("10 memory", req("GET", "/api/memory", auth=T)[0] == 200)

    failed = [n for n, ok in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()