import subprocess
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class LSPDocument:
    uri: str
    content: str
    version: int = 1


class DenoLSP:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._callbacks: dict[int, callable] = {}

    def start(self):
        self.process = subprocess.Popen(
            ["deno", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace_root,
        )

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def _send(self, method: str, params: dict):
        if not self.process:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        body = json.dumps(request).encode() + b"\n"
        self.process.stdin.write(body)
        self.process.stdin.flush()
        return self._request_id

    def _read_response(self, request_id: int):
        if not self.process:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        response = json.loads(line)
        if response.get("id") == request_id:
            return response.get("result")
        return None

    def initialize(self, root_uri: str):
        result = self._send(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "capabilities": {},
            },
        )
        if result:
            self._read_response(result)
        self._send("initialized", {})

    def get_definitions(self, file_path: str, line: int, character: int) -> list[dict]:
        uri = f"file://{file_path}"
        req_id = self._send(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        if req_id:
            return self._read_response(req_id) or []
        return []

    def get_references(self, file_path: str, line: int, character: int) -> list[dict]:
        uri = f"file://{file_path}"
        req_id = self._send(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": True},
            },
        )
        if req_id:
            return self._read_response(req_id) or []
        return []

    def get_type_definition(self, file_path: str, line: int, character: int) -> list[dict]:
        uri = f"file://{file_path}"
        req_id = self._send(
            "textDocument/typeDefinition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        if req_id:
            return self._read_response(req_id) or []
        return []
