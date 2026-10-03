import base64
import os
import threading
from itertools import cycle

import requests

from .utils import _DEFAULT_TIMEOUT_SECONDS, _ERROR_MSG_PREFIX, RunCodeResponse, RunStatus

# Default sandbox servers - can be overridden via environment variable or function parameter
DEFAULT_SANDBOX_SERVERS = [
    # "fs-mbz-gpu-044", # Add more servers here
]

# Thread-safe cycle iterator for round-robin load balancing
server_cycle = None
cycle_lock = threading.Lock()


def _parse_sandbox_servers(servers_input):
    """Parse sandbox servers from various input formats"""
    if not servers_input:
        return DEFAULT_SANDBOX_SERVERS

    if isinstance(servers_input, str):
        # Single server or comma-separated servers
        if "," in servers_input:
            return [server.strip() for server in servers_input.split(",")]
        else:
            return [servers_input.strip()]
    elif isinstance(servers_input, list):
        return servers_input
    else:
        raise ValueError(f"Invalid sandbox servers format: {type(servers_input)}. Expected str or list.")


def _get_next_server(server_cycle):
    """Get the next server in round-robin fashion thread-safely."""
    with cycle_lock:
        return next(server_cycle)


def code_exec_sandboxfusion(
    code,
    stdin: str = None,
    timeout=_DEFAULT_TIMEOUT_SECONDS,
    pytest: str = None,
    solution: str = None,
    sandbox_servers=None,
    memory_limit_mb: int = 1024,
):
    """
    Execute Python code using SandboxFusion remote service.

    Args:
        code: Python code to execute
        stdin: Optional input to pass to the code
        timeout: Timeout in seconds (default from utils)
        pytest: Optional pytest code that imports the submitted code from solution.py
        solution: Optional standalone test script that imports the submitted code from solution.py
        sandbox_servers: Optional server names for sandbox servers. Can be:
                        - Single server string: "fs-mbz-gpu-044"
                        - Comma-separated servers: "fs-mbz-gpu-044,fs-mbz-gpu-045"
                        - List of servers: ["fs-mbz-gpu-044", "fs-mbz-gpu-045"]
                        - None: Uses SANDBOX_FUSION_SERVERS environment variable or default

    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        if pytest is not None and solution is not None:
            raise ValueError("pytest and solution cannot be used together")
        if stdin is not None and (pytest is not None or solution is not None):
            raise ValueError("stdin is not supported with pytest or solution")

        # Determine sandbox servers to use
        if sandbox_servers is None:
            sandbox_servers = os.getenv("SANDBOX_FUSION_SERVERS", "")

        servers = _parse_sandbox_servers(sandbox_servers)
        server_cycle = cycle(servers)

        if not servers:
            return (
                False,
                _ERROR_MSG_PREFIX
                + "No sandbox servers configured. Set SANDBOX_FUSION_SERVERS environment variable or pass sandbox_servers parameter.",
            )

        language = "python"
        files = {}
        request_code = code
        if pytest is not None:
            language = "pytest"
            files["solution.py"] = base64.b64encode(code.encode()).decode()
            request_code = pytest
        elif solution is not None:
            files["solution.py"] = base64.b64encode(code.encode()).decode()
            request_code = solution

        request_data = {
            "language": language,
            "code": request_code,
            "stdin": stdin,
            "run_timeout": timeout,
            "memory_limit_MB": memory_limit_mb,
            "files": files,
        }

        # Try each server (for load balancing/failover)
        for _ in range(len(servers)):
            try:
                server = _get_next_server(server_cycle)
                url = f"http://{server}:8080/run_code"
                response = requests.post(url, json=request_data, timeout=timeout + 2)

                if response.status_code != 200:
                    continue  # Try next server

                result = RunCodeResponse(**response.json())
                if result.status == RunStatus.Success:
                    return True, result.run_result.stdout or ""
                else:
                    stdout = result.run_result.stdout if result.run_result else ""
                    stderr = result.run_result.stderr if result.run_result else result.message
                    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout or ''}\n\nSTDERR:\n{stderr or ''}"

            except requests.exceptions.RequestException:
                continue  # Try next server

        # If we get here, all servers failed
        return False, _ERROR_MSG_PREFIX + f"All sandbox servers failed to process the request. Servers tried: {servers}"

    except Exception as e:
        return False, _ERROR_MSG_PREFIX + f"Execution error: {str(e)}"


def code_exec_sandboxfusion_with_pytest(code, pytest_code, timeout=_DEFAULT_TIMEOUT_SECONDS, sandbox_servers=None):
    """
    Execute Python code with pytest using SandboxFusion remote service.

    Args:
        code: Python solution code
        pytest_code: Pytest test code
        timeout: Timeout in seconds
        sandbox_servers: Optional server names for sandbox servers (same format as code_exec_sandboxfusion)

    Returns:
        tuple: (success: bool, output: str)
    """
    return code_exec_sandboxfusion(
        code,
        pytest=pytest_code,
        timeout=timeout,
        sandbox_servers=sandbox_servers,
    )
