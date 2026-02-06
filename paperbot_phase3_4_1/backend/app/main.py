from __future__ import annotations

import asyncio
import logging
from importlib import resources

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .utils.logging_compat import setup_logging
from .binance.rest import BinanceRest
from .binance.ws_market import BinanceMarketWS
from .bot.runner import BotRunner
from .config import Settings
from .state.store import Store

setup_logging("INFO")
log = logging.getLogger("app.main")


def create_app() -> FastAPI:
    s = Settings()
    store = Store(s)
    rest = BinanceRest(s)
    ws = BinanceMarketWS(s)
    runner = BotRunner(store, rest, ws, s)

    app = FastAPI(title="Binance USD-M TESTNET Paper Bot")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = s
    app.state.store = store
    app.state.rest = rest
    app.state.ws = ws
    app.state.runner = runner

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            tick = await rest.get_tick_size(s.SYMBOL)
            await store.set_tick_size(tick)
            log.info("tick_size symbol=%s tick=%s", s.SYMBOL, tick)
        except Exception as e:
            log.warning("tick_size_fetch_failed err=%s", e)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        try:
            await runner.stop()
        except Exception:
            pass
        try:
            await rest.close()
        except Exception:
            pass

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = resources.files("backend.app.web").joinpath("index.html")
        html = html_path.read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/snapshot")
    async def api_snapshot() -> JSONResponse:
        return JSONResponse(await store.snapshot())

    @app.post("/api/bot/start")
    async def api_start() -> JSONResponse:
        await runner.start()
        return JSONResponse({"ok": True})

    @app.post("/api/bot/stop")
    async def api_stop() -> JSONResponse:
        await runner.stop()
        return JSONResponse({"ok": True})

    @app.post("/api/bot/reset")
    async def api_reset() -> JSONResponse:
        await runner.reset()
        return JSONResponse({"ok": True})

    @app.post("/api/bot/clear_kill")
    async def api_clear_kill() -> JSONResponse:
        await store.clear_kill()
        return JSONResponse({"ok": True})

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(sock: WebSocket) -> None:
        await sock.accept()
        try:
            while True:
                snap = await store.snapshot()
                await sock.send_json(snap)
                await asyncio.sleep(s.DASH_PUSH_MS / 1000.0)
        except WebSocketDisconnect:
            return
        except Exception as e:
            log.warning("dashboard_ws_error err=%s", e)
            return

    return app


app = create_app()

