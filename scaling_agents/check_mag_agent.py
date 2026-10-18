"""Check that this machine can reach MAG and run one agent through it (hello world).

Run it on any cluster (e.g. Kestrel) before submitting the GINO optimizer agents:

    source kestrel_env_setup.sh        # or export the ANTHROPIC_* variables yourself
    python scaling_agents/check_mag_agent.py

Each step prints PASS or FAIL with the reason:
  1. the MAG settings are in the environment (token, base URL, provider header);
  2. MAG answers GET /v1/models with the token, and lists the Claude models it serves;
  3. MAG answers one direct POST /v1/messages to the model, so a bad token, model name or provider
     shows up in seconds with MAG's own error (the agent CLI would retry it for minutes);
  4. one Academy agent runs one Claude Agent SDK query through MAG, with one tool call.

The model is GINO_LLM_MODEL (default claude-opus-5-5), as in 4_gino_optimizer_board.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

MODEL = os.environ.get('GINO_LLM_MODEL', 'claude-opus-5-5')
# Same Claude Code settings the GINO agents use (MAG's recommendations).
CLI_ENV = {
    'DISABLE_NON_ESSENTIAL_MODEL_CALLS': '1',
    'DISABLE_TELEMETRY': '1',
    'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '16384',
}
AGENT_TIMEOUT_S = 180
MAGIC = 'kestrel-says-hello'  # only the tool knows it, so the reply proves the tool call happened


def report(ok: bool, step: str, detail: str) -> bool:
    print(f'[{"PASS" if ok else "FAIL"}] {step}: {detail}', flush=True)
    return ok


def check_env() -> bool:
    token = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
    base = os.environ.get('ANTHROPIC_BASE_URL', '')
    headers = os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '')
    missing = [name for name, value in (('ANTHROPIC_AUTH_TOKEN', token), ('ANTHROPIC_BASE_URL', base))
               if not value or '<' in value]
    if missing:
        return report(False, '1 environment', f'{", ".join(missing)} not set (or still a placeholder)')
    proxies = {k: os.environ[k] for k in ('HTTPS_PROXY', 'https_proxy') if os.environ.get(k)}
    return report(True, '1 environment', f'base URL {base}, token {len(token)} chars, '
                  f'headers {headers or "(none)"}, proxy {proxies or "(none)"}, model {MODEL}')


def mag(path: str, payload: dict[str, Any] | None = None) -> Any:
    """GET (or POST payload) to MAG with the same auth and headers the Claude CLI sends."""
    base = os.environ['ANTHROPIC_BASE_URL'].rstrip('/')
    headers = {
        'Authorization': f'Bearer {os.environ["ANTHROPIC_AUTH_TOKEN"]}',
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        **dict(h.split(': ', 1) for h in os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '').splitlines() if ': ' in h),
    }
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f'{base}{path}', data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def http_failure(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f'HTTP {exc.code} {exc.reason}: {exc.read()[:400].decode(errors="replace")}'
    return f'cannot reach {os.environ["ANTHROPIC_BASE_URL"]} ({exc}); check the network'


def check_models() -> bool:
    try:
        body = mag('/v1/models')
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return report(False, '2 MAG /v1/models', http_failure(exc))
    entries = body.get('data', body if isinstance(body, list) else [])
    ids = sorted(str(m.get('id', m)) if isinstance(m, dict) else str(m) for m in entries)
    claude = [i for i in ids if 'claude' in i.lower()]
    found = MODEL in ids
    detail = f'{len(ids)} models; Claude: {", ".join(claude) or "(none)"}'
    if not found:
        detail += f' | WARNING: {MODEL} is not listed; set GINO_LLM_MODEL to one of the names above'
    return report(True, '2 MAG /v1/models', detail)


def check_message() -> bool:
    start = time.time()
    try:
        body = mag('/v1/messages', {
            'model': MODEL, 'max_tokens': 20,
            'messages': [{'role': 'user', 'content': 'Reply with exactly: hello world'}],
        })
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return report(False, f'3 MAG message to {MODEL}', http_failure(exc))
    text = ''.join(block.get('text', '') for block in body.get('content', []) if isinstance(block, dict))
    return report(bool(text), f'3 MAG message to {MODEL}',
                  f'{time.time() - start:.1f} s, served by {body.get("model", "?")}, reply {text.strip()[:100]!r}')


async def check_agent() -> bool:
    from academy.agent import action
    from academy.agent import Agent
    from academy.exchange import LocalExchangeFactory
    from academy.manager import Manager
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import query
    from claude_agent_sdk import ResultMessage
    from claude_agent_sdk import tool

    class HelloAgent(Agent):
        @action
        async def hello(self) -> dict[str, Any]:
            calls: list[str] = []

            @tool('secret_word', 'Returns the secret word for this check.', {})
            async def secret_word(args: dict[str, Any]) -> dict[str, Any]:
                calls.append('secret_word')
                return {'content': [{'type': 'text', 'text': MAGIC}]}

            options = ClaudeAgentOptions(
                model=MODEL,
                system_prompt='You are a connectivity check. Be brief.',
                mcp_servers={'check': create_sdk_mcp_server('check', tools=[secret_word])},
                tools=[],
                strict_mcp_config=True,
                allowed_tools=['mcp__check__secret_word'],
                setting_sources=[],
                env=CLI_ENV,
                max_turns=4,
            )
            result: dict[str, Any] = {'text': '', 'error': None}
            async for message in query(
                prompt='Call the secret_word tool, then reply exactly: hello world, the secret word is <word>.',
                options=options,
            ):
                if isinstance(message, ResultMessage):
                    error = None
                    if message.is_error:
                        error = f'{message.subtype} (API status {message.api_error_status}, {message.errors})'
                    result.update(text=message.result or '', error=error)
            result['tool_calls'] = calls
            return result

    start = time.time()
    async with await Manager.from_exchange_factory(factory=LocalExchangeFactory()) as manager:
        handle = await manager.launch(HelloAgent)
        result = await asyncio.wait_for(handle.hello(), timeout=AGENT_TIMEOUT_S)
    seconds = time.time() - start
    if result['error']:
        return report(False, '4 agent via MAG', f'query ended with {result["error"]}: {result["text"][:300]!r}')
    ok = MAGIC in result['text'] and result['tool_calls'] == ['secret_word']
    detail = f'{seconds:.0f} s, tool calls {result["tool_calls"]}, reply {result["text"].strip()[:200]!r}'
    if not ok:
        detail += ' | the reply or the tool call is missing: the model answered but tool use did not work'
    return report(ok, '4 agent via MAG', detail)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    if not check_env():
        return 1
    # The model listing is informative only, so a failure there doesn't stop the real tests.
    models_ok = check_models()
    if not check_message():
        print('MAG rejected a direct request (reason above); the agent would only retry it, so stopping.')
        return 1
    try:
        agent_ok = asyncio.run(check_agent())
    except TimeoutError:
        agent_ok = report(False, '4 agent via MAG', f'no reply in {AGENT_TIMEOUT_S} s although the direct '
                          'request worked; the Claude CLI is probably retrying, so check its network access')
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic; show whatever failed
        agent_ok = report(False, '4 agent via MAG', f'{type(exc).__name__}: {exc}')
    if agent_ok and models_ok:
        print('All checks passed: this machine can run the GINO optimizer agents.', flush=True)
    elif agent_ok:
        print('The agent works through MAG; only the model listing failed (see above).', flush=True)
    else:
        print('Agent check failed; see the reason above.', flush=True)
    return 0 if agent_ok else 1


if __name__ == '__main__':
    sys.exit(main())
