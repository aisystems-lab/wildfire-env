import asyncio
import json
import mimetypes
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve


class ViewerServer:
    def __init__(self, static_dir: Path, host: str = "127.0.0.1", port: int = 8765):
        self.static_dir = Path(static_dir)
        self.host = host
        self.port = port

        self._clients: set[WebSocketServerProtocol] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_terrain: Optional[dict[str, Any]] = None
        self._latest_snapshot: Optional[dict[str, Any]] = None

    @property
    def viewer_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.static_dir.exists():
            raise FileNotFoundError(
                f"Packaged 3D viewer assets not found at {self.static_dir}. Run `npm run build` in `web/` first."
            )

        self._started.clear()
        self._thread = threading.Thread(target=self._run, name="firecastrl-viewer", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)
        if not self._started.is_set():
            raise RuntimeError("Timed out starting 3D viewer server")

    def stop(self) -> None:
        if not self._loop:
            return

        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        future.result(timeout=5)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._server = None
        self._clients.clear()

    def publish_terrain(self, payload: dict[str, Any]) -> None:
        self.start()
        with self._state_lock:
            self._latest_terrain = payload
        self._broadcast(payload)

    def publish_snapshot(self, payload: dict[str, Any]) -> None:
        self.start()
        with self._state_lock:
            self._latest_snapshot = payload
        self._broadcast(payload)

    def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_payload(payload), self._loop)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        self._server = loop.run_until_complete(
            serve(
                self._handler,
                self.host,
                self.port,
                process_request=self._process_request,
                max_size=None,
            )
        )
        self._started.set()

        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._finalize_shutdown())
            loop.close()

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        if websocket.path != "/ws":
            await websocket.close()
            return

        self._clients.add(websocket)
        try:
            try:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=5)
                message = json.loads(raw_message)
                if message.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
            except (asyncio.TimeoutError, json.JSONDecodeError, TypeError):
                pass

            with self._state_lock:
                terrain = self._latest_terrain
                snapshot = self._latest_snapshot

            if terrain is not None:
                await websocket.send(json.dumps(terrain))
            if snapshot is not None:
                await websocket.send(json.dumps(snapshot))

            async for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if message.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
        except ConnectionClosed:
            return
        finally:
            self._clients.discard(websocket)

    async def _broadcast_payload(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return

        message = json.dumps(payload)
        stale_clients = []
        for client in list(self._clients):
            try:
                await client.send(message)
            except ConnectionClosed:
                stale_clients.append(client)

        for client in stale_clients:
            self._clients.discard(client)

    async def _shutdown(self) -> None:
        await self._finalize_shutdown()
        if self._loop and self._loop.is_running():
            self._loop.stop()

    async def _finalize_shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._clients:
            await asyncio.gather(*(client.close() for client in list(self._clients)), return_exceptions=True)
            self._clients.clear()

    async def _process_request(self, path: str, _request_headers):
        parsed = urlparse(path)
        if parsed.path == "/ws":
            return None

        target = self._resolve_static_path(parsed.path)
        if target is None:
            return self._response(HTTPStatus.NOT_FOUND, b"Not found")

        data = target.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(target))
        headers = [
            ("Content-Type", mime_type or "application/octet-stream"),
            ("Content-Length", str(len(data))),
            ("Cache-Control", "no-cache"),
        ]
        return HTTPStatus.OK, headers, data

    def _resolve_static_path(self, request_path: str) -> Optional[Path]:
        clean_path = unquote(request_path.split("?", 1)[0])
        if clean_path in {"", "/"}:
            return self.static_dir / "index.html"

        relative_path = clean_path.lstrip("/")
        candidate = (self.static_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.static_dir.resolve())
        except ValueError:
            return None

        if candidate.is_file():
            return candidate

        if "." not in Path(relative_path).name:
            fallback = self.static_dir / "index.html"
            if fallback.exists():
                return fallback

        return None

    def _response(self, status: HTTPStatus, body: bytes):
        return status, [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))], body
