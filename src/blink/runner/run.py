#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Blink development runner for Daily, WebRTC, and telephony workflows.

This development runner executes Blink bots and provides the supporting
infrastructure they need - creating Daily rooms and tokens, managing WebRTC
connections, and setting up telephony webhook/WebSocket infrastructure. It
supports multiple transport types with a unified interface.

Install with::

    pip install blink-ai[runner]

All bots must implement a `bot(runner_args)` async function as the entry point.
The server automatically discovers and executes this function when connections
are established.

Single transport example::

    async def bot(runner_args: RunnerArguments):
        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            "Bot",
            DailyParams(...)
        )
        # Your bot logic here
        await run_pipeline(transport)

    if __name__ == "__main__":
        from blink.runner.run import main
        main()

Multiple transport example::

    async def bot(runner_args: RunnerArguments):
        # Type-safe transport detection
        if isinstance(runner_args, DailyRunnerArguments):
            transport = setup_daily_transport(runner_args)  # Your application code
        elif isinstance(runner_args, SmallWebRTCRunnerArguments):
            transport = setup_webrtc_transport(runner_args)  # Your application code
        elif isinstance(runner_args, WebSocketRunnerArguments):
            transport = setup_telephony_transport(runner_args)  # Your application code

        # Your bot implementation
        await run_pipeline(transport)

Supported transports:

- Daily - Creates rooms and tokens, runs bot as participant
- WebRTC - Provides local WebRTC interface with the repo-owned Blink browser UI
- Telephony - Handles webhook and WebSocket connections for Twilio, Telnyx, Plivo, Exotel

To run locally:

- WebRTC: `python bot.py -t webrtc`
- ESP32: `python bot.py -t webrtc --esp32 --host 192.168.1.100`
- Daily (server): `python bot.py -t daily`
- Daily (direct, testing only): `python bot.py -d`
- Telephony: `python bot.py -t twilio -x your_username.ngrok.io`
- Exotel: `python bot.py -t exotel` (no proxy needed, but ngrok connection to HTTP 7860 is required)
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import os
import sys
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union, cast

import aiohttp
from loguru import logger

from blink.project_identity import PROJECT_IDENTITY, pip_install_command
from blink.runner.types import (
    DailyRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from blink.web.server_startup import STARTUP_TIMEOUT_SECS, start_uvicorn_server
from blink.web.smallwebrtc_ui import create_smallwebrtc_root_redirect, mount_smallwebrtc_ui

if TYPE_CHECKING:
    from fastapi import BackgroundTasks, FastAPI, Request, WebSocket

_RUNNER_IMPORT_ERROR: ImportError | None = None
_uvicorn = None
_FastAPI = None
_load_dotenv: Any = None
_Header = cast(Any, None)
_HTTPException = cast(Any, None)
_CORSMiddleware = cast(Any, None)
_FileResponse = cast(Any, None)
_HTMLResponse = cast(Any, None)
_RedirectResponse = cast(Any, None)
_Response = cast(Any, None)

try:
    import uvicorn as _uvicorn
    from dotenv import load_dotenv as _load_dotenv
    from fastapi import (
        BackgroundTasks as _BackgroundTasks,
    )
    from fastapi import (
        FastAPI as _FastAPI,
    )
    from fastapi import (
        Header as _Header,
    )
    from fastapi import (
        HTTPException as _HTTPException,
    )
    from fastapi import (
        Request as _Request,
    )
    from fastapi import (
        WebSocket as _WebSocket,
    )
    from fastapi.middleware.cors import CORSMiddleware as _CORSMiddleware
    from fastapi.responses import (
        FileResponse as _FileResponse,
    )
    from fastapi.responses import (
        HTMLResponse as _HTMLResponse,
    )
    from fastapi.responses import (
        RedirectResponse as _RedirectResponse,
    )
    from fastapi.responses import (
        Response as _Response,
    )
except ImportError as e:
    _RUNNER_IMPORT_ERROR = e
else:
    globals().update(
        BackgroundTasks=_BackgroundTasks,
        FastAPI=_FastAPI,
        Header=_Header,
        HTTPException=_HTTPException,
        Request=_Request,
        WebSocket=_WebSocket,
        CORSMiddleware=_CORSMiddleware,
        FileResponse=_FileResponse,
        HTMLResponse=_HTMLResponse,
        RedirectResponse=_RedirectResponse,
        Response=_Response,
    )


def load_dotenv(*args: Any, **kwargs: Any) -> bool:
    """Load a dotenv file when the optional dependency is available."""
    if _RUNNER_IMPORT_ERROR is not None:
        return False
    return _load_dotenv(*args, **kwargs)


def _require_runner_dependencies():
    """Raise a clean error when the optional runner extras are unavailable."""
    if _RUNNER_IMPORT_ERROR is None:
        return

    install_hint = pip_install_command("runner")
    message = f"Runner dependencies required. Install with: {install_hint}"
    logger.error(f"Runner dependencies not available: {_RUNNER_IMPORT_ERROR}")
    logger.error(f"To use {PROJECT_IDENTITY.display_name} runners, install with: {install_hint}")
    raise ImportError(message) from _RUNNER_IMPORT_ERROR


def _require_runner_app() -> FastAPI:
    """Return the module-level FastAPI app once runner deps are available."""
    _require_runner_dependencies()
    if app is None:
        raise RuntimeError("Runner HTTP application could not be initialized.")
    return app


load_dotenv(override=True)
os.environ["ENV"] = "local"

TELEPHONY_TRANSPORTS = ["twilio", "telnyx", "plivo", "exotel"]

RUNNER_DOWNLOADS_FOLDER: str | None = None
RUNNER_HOST: str = "localhost"
RUNNER_PORT: int = 7860

app: FastAPI | None = _FastAPI() if _RUNNER_IMPORT_ERROR is None and _FastAPI is not None else None
"""The FastAPI application instance.

Import this to add custom routes from other packages before calling
:func:`main`::

    from blink.runner.run import app, main

    @app.get("/my-route")
    async def my_route():
        return {"hello": "world"}

    if __name__ == "__main__":
        main()
"""


def _get_bot_module():
    """Get the bot module from the calling script."""
    import importlib.util

    # Get the main module (the file that was executed)
    main_module = sys.modules["__main__"]

    # Check if it has a bot function
    if hasattr(main_module, "bot"):
        return main_module

    # Try to import 'bot' module from current directory
    try:
        import bot  # type: ignore[import-untyped]

        return bot
    except ImportError:
        pass

    # Look for any .py file in current directory that has a bot function
    # (excluding server.py).
    cwd = os.getcwd()
    for filename in os.listdir(cwd):
        if filename.endswith(".py") and filename != "server.py":
            try:
                module_name = filename[:-3]  # Remove .py extension
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(cwd, filename)
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "bot"):
                    return module
            except Exception:
                continue

    raise ImportError(
        "Could not find 'bot' function. Make sure your bot file has a 'bot' function."
    )


async def _run_telephony_bot(websocket: WebSocket, args: argparse.Namespace):
    """Run a bot for telephony transports."""
    bot_module = _get_bot_module()

    # Just pass the WebSocket - let the bot handle parsing
    runner_args = WebSocketRunnerArguments(websocket=websocket)
    runner_args.cli_args = args

    await bot_module.bot(runner_args)


def _configure_server_app(args: argparse.Namespace):
    """Configure the module-level FastAPI app with transport-specific routes."""
    runner_app = _require_runner_app()

    runner_app.add_middleware(
        _CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set up transport-specific routes
    if args.transport == "webrtc":
        _setup_webrtc_routes(runner_app, args)
        if args.whatsapp:
            _setup_whatsapp_routes(runner_app, args)
    elif args.transport == "daily":
        _setup_daily_routes(runner_app, args)
    elif args.transport in TELEPHONY_TRANSPORTS:
        _setup_telephony_routes(runner_app, args)
    else:
        logger.warning(f"Unknown transport type: {args.transport}")


def _setup_webrtc_routes(app: FastAPI, args: argparse.Namespace):
    """Set up WebRTC-specific routes."""
    try:
        from blink.transports.smallwebrtc.connection import SmallWebRTCConnection
        from blink.transports.smallwebrtc.request_handler import (
            IceCandidate,
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
    except ImportError as e:
        logger.error(f"WebRTC transport dependencies not installed: {e}")
        return

    class IceServer(TypedDict, total=False):
        urls: Union[str, list[str]]

    class IceConfig(TypedDict):
        iceServers: list[IceServer]

    class StartBotResult(TypedDict, total=False):
        sessionId: str
        iceConfig: Optional[IceConfig]

    # In-memory store of active sessions: session_id -> session info
    active_sessions: dict[str, dict[str, Any]] = {}

    # The mounted /client UI is the repo-owned Blink browser bundle.
    mount_smallwebrtc_ui(app)
    app.add_api_route(
        "/", create_smallwebrtc_root_redirect(), methods=["GET"], include_in_schema=False
    )

    @app.get("/files/{filename:path}")
    async def download_file(filename: str):
        """Handle file downloads."""
        if not args.folder:
            logger.warning(f"Attempting to dowload {filename}, but downloads folder not setup.")
            return

        file_path = Path(args.folder) / filename
        if not os.path.exists(file_path):
            raise _HTTPException(404)

        media_type, _ = mimetypes.guess_type(file_path)

        return _FileResponse(path=file_path, media_type=media_type, filename=filename)

    # Initialize the SmallWebRTC request handler
    small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler(
        esp32_mode=args.esp32, host=args.host
    )

    @app.post("/api/offer")
    async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
        """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""

        # Prepare runner arguments with the callback to run your bot
        async def webrtc_connection_callback(connection: SmallWebRTCConnection):
            bot_module = _get_bot_module()

            runner_args = SmallWebRTCRunnerArguments(
                webrtc_connection=connection, body=request.request_data
            )
            runner_args.cli_args = args
            background_tasks.add_task(bot_module.bot, runner_args)

        # Delegate handling to SmallWebRTCRequestHandler
        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
        return answer

    @app.patch("/api/offer")
    async def ice_candidate(request: SmallWebRTCPatchRequest):
        """Handle WebRTC new ice candidate requests."""
        logger.debug(f"Received patch request: {request}")
        await small_webrtc_handler.handle_patch_request(request)
        return {"status": "success"}

    @app.post("/start")
    async def rtvi_start(request: Request):
        """Mimic the legacy cloud-compatible ``/start`` endpoint."""
        # Parse the request body
        try:
            request_data = await request.json()
            logger.debug(f"Received request: {request_data}")
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            request_data = {}

        # Store session info immediately in memory, matching the legacy cloud session contract.
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = request_data.get("body", {})

        result: StartBotResult = {"sessionId": session_id}
        if request_data.get("enableDefaultIceServers"):
            result["iceConfig"] = IceConfig(
                iceServers=[IceServer(urls=["stun:stun.l.google.com:19302"])]
            )

        return result

    @app.api_route(
        "/sessions/{session_id}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy_request(
        session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
    ):
        """Proxy legacy cloud-compatible session traffic."""
        active_session = active_sessions.get(session_id)
        if active_session is None:
            return _Response(content="Invalid or not-yet-ready session_id", status_code=404)

        if path.endswith("api/offer"):
            # Parse the request body and convert to SmallWebRTCRequest
            try:
                request_data = await request.json()
                if request.method == HTTPMethod.POST.value:
                    webrtc_request = SmallWebRTCRequest(
                        sdp=request_data["sdp"],
                        type=request_data["type"],
                        pc_id=request_data.get("pc_id"),
                        restart_pc=request_data.get("restart_pc"),
                        request_data=request_data.get("request_data")
                        or request_data.get("requestData")
                        or active_session,
                    )
                    return await offer(webrtc_request, background_tasks)
                elif request.method == HTTPMethod.PATCH.value:
                    patch_request = SmallWebRTCPatchRequest(
                        pc_id=request_data["pc_id"],
                        candidates=[IceCandidate(**c) for c in request_data.get("candidates", [])],
                    )
                    return await ice_candidate(patch_request)
            except Exception as e:
                logger.error(f"Failed to parse WebRTC request: {e}")
                return _Response(content="Invalid WebRTC request", status_code=400)

        logger.info(f"Received request for path: {path}")
        return _Response(status_code=200)

    @asynccontextmanager
    async def smallwebrtc_lifespan(app: FastAPI):
        """Manage FastAPI application lifecycle and cleanup connections."""
        yield
        await small_webrtc_handler.close()

    # Add the SmallWebRTC lifespan to the app
    _add_lifespan_to_app(app, smallwebrtc_lifespan)


def _add_lifespan_to_app(app: FastAPI, new_lifespan):
    """Add a new lifespan context manager to the app, combining with existing if present.

    Args:
        app: The FastAPI application instance
        new_lifespan: The new lifespan context manager to add
    """
    if hasattr(app.router, "lifespan_context") and app.router.lifespan_context is not None:
        # If there's already a lifespan context, combine them
        existing_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def combined_lifespan(app: FastAPI):
            async with existing_lifespan(app):
                async with new_lifespan(app):
                    yield

        app.router.lifespan_context = combined_lifespan
    else:
        # No existing lifespan, use the new one
        app.router.lifespan_context = new_lifespan


def _setup_whatsapp_routes(app: FastAPI, args: argparse.Namespace):
    """Set up WhatsApp-specific routes."""
    whatsapp_app_secret = os.getenv("WHATSAPP_APP_SECRET")
    whatsapp_phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_token = os.getenv("WHATSAPP_TOKEN")
    whatsapp_webhook_verification_token = os.getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN")

    if not all(
        [
            whatsapp_app_secret,
            whatsapp_phone_number_id,
            whatsapp_token,
            whatsapp_webhook_verification_token,
        ]
    ):
        logger.error(
            """Missing required environment variables for WhatsApp transport:
    WHATSAPP_APP_SECRET
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_TOKEN
    WHATSAPP_WEBHOOK_VERIFICATION_TOKEN
            """
        )
        return

    try:
        from blink.transports.smallwebrtc.connection import SmallWebRTCConnection
        from blink.transports.whatsapp.api import WhatsAppWebhookRequest
        from blink.transports.whatsapp.client import WhatsAppClient
    except ImportError as e:
        logger.error(f"WhatsApp transport dependencies not installed: {e}")
        return

    # Global WhatsApp client instance
    whatsapp_client: Optional[WhatsAppClient] = None
    assert (
        whatsapp_app_secret
        and whatsapp_phone_number_id
        and whatsapp_token
        and whatsapp_webhook_verification_token
    )

    @app.get(
        "/whatsapp",
        summary="Verify WhatsApp webhook",
        description="Handles WhatsApp webhook verification requests from Meta",
    )
    async def verify_webhook(request: Request):
        """Verify WhatsApp webhook endpoint.

        This endpoint is called by Meta's WhatsApp Business API to verify
        the webhook URL during setup. It validates the verification token
        and returns the challenge parameter if successful.
        """
        if whatsapp_client is None:
            logger.error("WhatsApp client is not initialized")
            raise _HTTPException(status_code=503, detail="Service unavailable")

        params = dict(request.query_params)
        logger.debug(f"Webhook verification request received with params: {list(params.keys())}")

        try:
            result = await whatsapp_client.handle_verify_webhook_request(
                params=params,
                expected_verification_token=whatsapp_webhook_verification_token,
            )
            logger.info("Webhook verification successful")
            return result
        except ValueError as e:
            logger.warning(f"Webhook verification failed: {e}")
            raise _HTTPException(status_code=403, detail="Verification failed")

    @app.post(
        "/whatsapp",
        summary="Handle WhatsApp webhook events",
        description="Processes incoming WhatsApp messages and call events",
    )
    async def whatsapp_webhook(
        body: WhatsAppWebhookRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        x_hub_signature_256: str = _Header(None),
    ):
        """Handle incoming WhatsApp webhook events.

        For call events, establishes WebRTC connections and spawns bot instances
        in the background to handle real-time communication.
        """
        if whatsapp_client is None:
            logger.error("WhatsApp client is not initialized")
            raise _HTTPException(status_code=503, detail="Service unavailable")

        # Validate webhook object type
        if body.object != "whatsapp_business_account":
            logger.warning(f"Invalid webhook object type: {body.object}")
            raise _HTTPException(status_code=400, detail="Invalid object type")

        logger.debug(f"Processing WhatsApp webhook: {body.model_dump()}")

        async def connection_callback(connection: SmallWebRTCConnection):
            """Handle new WebRTC connections from WhatsApp calls.

            Called when a WebRTC connection is established for a WhatsApp call.
            Spawns a bot instance to handle the conversation.

            Args:
                connection: The established WebRTC connection
            """
            bot_module = _get_bot_module()
            runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
            runner_args.cli_args = args
            background_tasks.add_task(bot_module.bot, runner_args)

        try:
            # Process the webhook request
            raw_body = await request.body()
            result = await whatsapp_client.handle_webhook_request(
                body, connection_callback, sha256_signature=x_hub_signature_256, raw_body=raw_body
            )
            logger.debug(f"Webhook processed successfully: {result}")
            return {"status": "success", "message": "Webhook processed successfully"}
        except ValueError as ve:
            logger.warning(f"Invalid webhook request format: {ve}")
            raise _HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
        except Exception as e:
            logger.error(f"Internal error processing webhook: {e}")
            raise _HTTPException(status_code=500, detail="Internal server error processing webhook")

    @asynccontextmanager
    async def whatsapp_lifespan(app: FastAPI):
        """Manage WhatsApp client lifecycle and cleanup connections."""
        nonlocal whatsapp_client

        # Initialize WhatsApp client with persistent HTTP session
        async with aiohttp.ClientSession() as session:
            whatsapp_client = WhatsAppClient(
                whatsapp_token=whatsapp_token,
                whatsapp_secret=whatsapp_app_secret,
                phone_number_id=whatsapp_phone_number_id,
                session=session,
            )
            logger.info("WhatsApp client initialized successfully")

            try:
                yield  # Run the application
            finally:
                # Cleanup all active calls on shutdown
                logger.info("Cleaning up WhatsApp client resources...")
                if whatsapp_client:
                    await whatsapp_client.terminate_all_calls()
                logger.info("WhatsApp cleanup completed")

    # Add the WhatsApp lifespan to the app
    _add_lifespan_to_app(app, whatsapp_lifespan)


def _setup_daily_routes(app: FastAPI, args: argparse.Namespace):
    """Set up Daily-specific routes."""

    @app.get("/")
    async def create_room_and_start_agent():
        """Launch a Daily bot and redirect to room."""
        print("Starting bot with Daily transport and redirecting to Daily room")

        import aiohttp

        from blink.runner.daily import configure

        async with aiohttp.ClientSession() as session:
            room_url, token = await configure(session)

            # Start the bot in the background with empty body for GET requests
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(room_url=room_url, token=token)
            runner_args.cli_args = args
            asyncio.create_task(bot_module.bot(runner_args))
            return _RedirectResponse(room_url)

    @app.post("/start")
    async def start_agent(request: Request):
        """Handler for /start endpoints.

        Expects POST body like::
            {
                "createDailyRoom": true,
                "dailyRoomProperties": { "start_video_off": true },
                "dailyMeetingTokenProperties": { "is_owner": true, "user_name": "Bot" },
                "body": { "custom_data": "value" }
            }
        """
        print("Starting bot with Daily transport")

        # Parse the request body
        try:
            request_data = await request.json()
            logger.debug(f"Received request: {request_data}")
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            request_data = {}

        create_daily_room = request_data.get("createDailyRoom", False)
        body = request_data.get("body", {})
        daily_room_properties_dict = request_data.get("dailyRoomProperties", None)
        daily_token_properties_dict = request_data.get("dailyMeetingTokenProperties", None)

        bot_module = _get_bot_module()

        existing_room_url = os.getenv("DAILY_ROOM_URL")

        result = None

        # Configure room if:
        # 1. Explicitly requested via createDailyRoom in payload
        # 2. Using pre-configured room from DAILY_ROOM_URL env var
        if create_daily_room or existing_room_url:
            import aiohttp

            from blink.runner.daily import configure
            from blink.transports.daily.utils import (
                DailyMeetingTokenProperties,
                DailyRoomProperties,
            )

            async with aiohttp.ClientSession() as session:
                # Parse dailyRoomProperties if provided
                room_properties = None
                if daily_room_properties_dict:
                    try:
                        room_properties = DailyRoomProperties(**daily_room_properties_dict)
                        logger.debug(f"Using custom room properties: {room_properties}")
                    except Exception as e:
                        logger.error(f"Failed to parse dailyRoomProperties: {e}")
                        # Continue without custom properties

                # Parse dailyMeetingTokenProperties if provided
                token_properties = None
                if daily_token_properties_dict:
                    try:
                        token_properties = DailyMeetingTokenProperties(
                            **daily_token_properties_dict
                        )
                        logger.debug(f"Using custom token properties: {token_properties}")
                    except Exception as e:
                        logger.error(f"Failed to parse dailyMeetingTokenProperties: {e}")
                        # Continue without custom properties

                room_url, token = await configure(
                    session, room_properties=room_properties, token_properties=token_properties
                )
                runner_args = DailyRunnerArguments(room_url=room_url, token=token, body=body)
                result = {
                    "dailyRoom": room_url,
                    "dailyToken": token,
                    "sessionId": str(uuid.uuid4()),
                }
        else:
            runner_args = RunnerArguments(body=body)

        # Update CLI args.
        runner_args.cli_args = args

        # Start the bot in the background
        asyncio.create_task(bot_module.bot(runner_args))

        return result

    if args.dialin:

        @app.post("/daily-dialin-webhook")
        async def handle_dialin_webhook(request: Request):
            """Handle incoming Daily PSTN dial-in webhook.

            This endpoint mimics the legacy cloud-compatible dial-in webhook handler.
            It receives Daily webhook data, creates a SIP-enabled room, and starts the bot.

            Expected webhook payload::

                {
                    "From": "+15551234567",
                    "To": "+15559876543",
                    "callId": "uuid-call-id",
                    "callDomain": "uuid-call-domain",
                    "sipHeaders": {...}  // optional
                }

            Returns::

                {
                    "dailyRoom": "https://...",
                    "dailyToken": "...",
                    "sessionId": "uuid"
                }
            """
            logger.debug("Received Daily dial-in webhook")

            try:
                data = await request.json()
                logger.debug(f"Webhook data: {data}")
            except Exception as e:
                logger.error(f"Failed to parse webhook data: {e}")
                raise _HTTPException(status_code=400, detail="Invalid JSON payload")

            # Handle webhook verification test (sent by Daily when configuring webhook)
            if data.get("test") or data.get("Test"):
                logger.debug("Webhook verification test received")
                return {"status": "OK"}

            # Validate required fields
            if not all(key in data for key in ["From", "To", "callId", "callDomain"]):
                raise _HTTPException(
                    status_code=400,
                    detail="Missing required fields: From, To, callId, callDomain",
                )

            import aiohttp

            from blink.runner.daily import configure
            from blink.runner.types import DailyDialinRequest, DialinSettings

            # Create Daily room with SIP capabilities
            async with aiohttp.ClientSession() as session:
                try:
                    room_config = await configure(session, sip_caller_phone=data.get("From"))
                except Exception as e:
                    logger.error(f"Failed to create Daily room: {e}")
                    raise _HTTPException(
                        status_code=500, detail=f"Failed to create Daily room: {str(e)}"
                    )

            # Get Daily API URL from environment, fallback to production
            daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

            # Get Daily API key from environment
            daily_api_key = os.getenv("DAILY_API_KEY")
            if not daily_api_key:
                logger.error("DAILY_API_KEY not found in environment")
                raise _HTTPException(
                    status_code=500, detail="DAILY_API_KEY not configured on server"
                )

            # Prepare dial-in settings matching the legacy cloud payload structure.
            dialin_settings = DialinSettings(
                call_id=data.get("callId"),
                call_domain=data.get("callDomain"),
                To=data.get("To"),
                From=data.get("From"),
                sip_headers=data.get("sipHeaders"),
            )

            # Create request body matching the legacy cloud payload.
            request_body = DailyDialinRequest(
                dialin_settings=dialin_settings,
                daily_api_key=daily_api_key,
                daily_api_url=daily_api_url,
            )

            # Start bot with dial-in context
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(
                room_url=room_config.room_url,
                token=room_config.token,
                body=request_body.model_dump(),
            )
            runner_args.cli_args = args

            asyncio.create_task(bot_module.bot(runner_args))

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Return response matching the legacy cloud response format.
            return {
                "dailyRoom": room_config.room_url,
                "dailyToken": room_config.token,
                "sessionId": session_id,
            }


def _setup_telephony_routes(app: FastAPI, args: argparse.Namespace):
    """Set up telephony-specific routes."""
    # XML response templates (Exotel doesn't use XML webhooks)
    XML_TEMPLATES = {
        "twilio": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
        "telnyx": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
        "plivo": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{args.proxy}/ws</Stream>
</Response>""",
    }

    @app.post("/")
    async def start_call():
        """Handle telephony webhook and return XML response."""
        if args.transport == "exotel":
            # Exotel doesn't use POST webhooks - redirect to proper documentation
            logger.debug("POST Exotel endpoint - not used")
            return {
                "error": "Exotel doesn't use POST webhooks",
                "websocket_url": f"wss://{args.proxy}/ws",
                "note": "Configure the WebSocket URL above in your Exotel App Bazaar Voicebot Applet",
            }
        else:
            logger.debug(f"POST {args.transport.upper()} XML")
            xml_content = XML_TEMPLATES.get(args.transport, "<Response></Response>")
            return _HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connections for telephony."""
        await websocket.accept()
        logger.debug("WebSocket connection accepted")
        await _run_telephony_bot(websocket, args)

    @app.get("/")
    async def start_agent():
        """Simple status endpoint for telephony transports."""
        return {"status": f"Bot started with {args.transport}"}


async def _run_daily_direct(args: argparse.Namespace):
    """Run Daily bot with direct connection (no FastAPI server)."""
    try:
        from blink.runner.daily import configure
    except ImportError as e:
        logger.error("Daily transport dependencies not installed.")
        return

    logger.info("Running with direct Daily connection...")

    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(session)

        # Direct connections have no request body, so use empty dict
        runner_args = DailyRunnerArguments(room_url=room_url, token=token)
        runner_args.handle_sigint = True
        runner_args.cli_args = args

        # Get the bot module and run it directly
        bot_module = _get_bot_module()

        print(f"📞 Joining Daily room: {room_url}")
        print("   (Direct connection - no web server needed)")
        print()

        await bot_module.bot(runner_args)


def _validate_and_clean_proxy(proxy: str) -> str:
    """Validate and clean proxy hostname, removing protocol if present."""
    if not proxy:
        return proxy

    original_proxy = proxy

    # Strip common protocols
    if proxy.startswith(("http://", "https://")):
        proxy = proxy.split("://", 1)[1]
        logger.warning(
            f"Removed protocol from proxy URL. Using '{proxy}' instead of '{original_proxy}'. "
            f"The --proxy argument expects only the hostname (e.g., 'mybot.ngrok.io')."
        )

    # Remove trailing slashes
    proxy = proxy.rstrip("/")

    return proxy


def runner_downloads_folder() -> Optional[str]:
    """Returns the folder where files are stored for later download."""
    return RUNNER_DOWNLOADS_FOLDER


def runner_host() -> str:
    """Returns the host name of this runner."""
    return RUNNER_HOST


def runner_port() -> int:
    """Returns the port of this runner."""
    return RUNNER_PORT


def _runner_ready_path(args: argparse.Namespace) -> str | None:
    """Return the readiness path to probe for the selected transport mode."""
    if args.transport == "webrtc" and not args.esp32 and not args.whatsapp:
        return "/client/"
    return None


async def _serve_runner_http(args: argparse.Namespace) -> None:
    """Start the runner HTTP server and wait for real readiness."""
    runner_app = _require_runner_app()
    if _uvicorn is None:
        raise RuntimeError("uvicorn is not available.")

    ready_path = _runner_ready_path(args)
    base_url = f"http://{args.host}:{args.port}"

    print()
    if ready_path:
        print(f"Waiting for {base_url}{ready_path} to become ready...")
    else:
        print(f"Waiting for {base_url} to accept connections...")
    print()

    server = _uvicorn.Server(_uvicorn.Config(runner_app, host=args.host, port=args.port))
    serve_task = await start_uvicorn_server(
        server,
        host=args.host,
        port=args.port,
        ready_path=ready_path,
        timeout_secs=STARTUP_TIMEOUT_SECS,
    )

    print()
    if args.transport == "webrtc":
        if args.esp32:
            print("🚀 Bot ready! (ESP32 mode)")
        elif args.whatsapp:
            print("🚀 Bot ready! (WhatsApp)")
        else:
            print("🚀 Bot ready!")
        if ready_path:
            print(f"   → Open {base_url}{ready_path} in your browser")
        else:
            print(f"   → HTTP server listening at {base_url}")
        print()
    elif args.transport == "daily":
        print("🚀 Bot ready!")
        if args.dialin:
            print(f"   → Daily dial-in webhook: {base_url}/daily-dialin-webhook")
            print("   → Configure this URL in your Daily phone number settings")
        else:
            print(f"   → Open {base_url} in your browser to start a session")
        print()

    await serve_task


def main(parser: Optional[argparse.ArgumentParser] = None):
    """Start the Blink development runner.

    Parses command-line arguments and starts a FastAPI server configured
    for the specified transport type.

    The runner discovers and runs any ``bot(runner_args)`` function found in the
    calling module.

    Command-line arguments:
       - --host: Server host address (default: localhost) 879
       - --port: Server port (default: 7860)
       - -t/--transport: Transport type (daily, webrtc, twilio, telnyx, plivo, exotel)
       - -x/--proxy: Public proxy hostname for telephony webhooks
       - -d/--direct: Connect directly to Daily room (automatically sets transport to daily)
       - -f/--folder: Path to downloads folder
       - --dialin: Enable Daily PSTN dial-in webhook handling (requires Daily transport)
       - --esp32: Enable SDP munging for ESP32 compatibility (requires --host with IP address)
       - --whatsapp: Ensure requried WhatsApp environment variables are present
       - -v/--verbose: Increase logging verbosity

    Args:
        parser: Optional custom argument parser. If provided, default runner
            arguments are added to it so bots can define their own CLI
            arguments. Custom arguments should not conflict with the default
            ones. Custom args are accessible via `runner_args.cli_args`.

    """
    global RUNNER_DOWNLOADS_FOLDER, RUNNER_HOST, RUNNER_PORT
    _require_runner_dependencies()

    if not parser:
        parser = argparse.ArgumentParser(
            description=f"{PROJECT_IDENTITY.display_name} Development Runner"
        )
    parser.add_argument("--host", type=str, default=RUNNER_HOST, help="Host address")
    parser.add_argument("--port", type=int, default=RUNNER_PORT, help="Port number")
    parser.add_argument(
        "-t",
        "--transport",
        type=str,
        choices=["daily", "webrtc", *TELEPHONY_TRANSPORTS],
        default="webrtc",
        help="Transport type",
    )
    parser.add_argument("-x", "--proxy", help="Public proxy host name")
    parser.add_argument(
        "-d",
        "--direct",
        action="store_true",
        default=False,
        help="Connect directly to Daily room (automatically sets transport to daily)",
    )
    parser.add_argument("-f", "--folder", type=str, help="Path to downloads folder")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    parser.add_argument(
        "--dialin",
        action="store_true",
        default=False,
        help="Enable Daily PSTN dial-in webhook handling (requires Daily transport)",
    )
    parser.add_argument(
        "--esp32",
        action="store_true",
        default=False,
        help="Enable SDP munging for ESP32 compatibility (requires --host with IP address)",
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        default=False,
        help="Ensure requried WhatsApp environment variables are present",
    )

    args = parser.parse_args()

    # Validate and clean proxy hostname
    if args.proxy:
        args.proxy = _validate_and_clean_proxy(args.proxy)

    # Auto-set transport to daily if --direct is used without explicit transport
    if args.direct and args.transport == "webrtc":  # webrtc is the default
        args.transport = "daily"
    elif args.direct and args.transport != "daily":
        logger.error("--direct flag only works with Daily transport (-t daily)")
        return

    # Validate ESP32 requirements
    if args.esp32 and args.host == "localhost":
        logger.error("For ESP32, you need to specify `--host IP` so we can do SDP munging.")
        return

    # Validate dial-in requirements
    if args.dialin and args.transport != "daily":
        logger.error("--dialin flag only works with Daily transport (-t daily)")
        return

    # Log level
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    # Handle direct Daily connection (no FastAPI server)
    if args.direct:
        print()
        print("🚀 Connecting directly to Daily room...")
        print()

        # Run direct Daily connection
        asyncio.run(_run_daily_direct(args))
        return

    RUNNER_DOWNLOADS_FOLDER = args.folder
    RUNNER_HOST = args.host
    RUNNER_PORT = args.port

    try:
        _configure_server_app(args)
    except Exception as exc:
        logger.error(f"Failed to configure the runner HTTP app: {exc}")
        raise

    try:
        asyncio.run(_serve_runner_http(args))
    except Exception as exc:
        logger.error(f"Runner HTTP startup failed: {exc}")
        raise


if __name__ == "__main__":
    main()
