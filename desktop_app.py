#!/usr/bin/env python3
"""
🧠 Personal LLM — Desktop Application Launcher
Launches a native desktop window with the Gradio-powered AI assistant.
Handles: dynamic ports, splash screen, graceful shutdown, model detection.
"""

import sys
import os
import socket
import time
import threading
import logging

# Ensure personal_llm package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find a free port on localhost by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_for_server(host: str, port: int, timeout: int = 30) -> bool:
    """
    Poll the server until it's ready, with a timeout.
    Returns True if server is ready, False if timed out.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((host, port))
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def check_models_exist() -> bool:
    """Check if there are any GGUF models available."""
    from personal_llm import config
    if not config.MODELS_DIR.exists():
        return False
    models = list(config.MODELS_DIR.glob("*.gguf"))
    return len(models) > 0


def show_error_dialog(title: str, message: str):
    """Show a native Windows error dialog using ctypes."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)  # MB_ICONERROR
    except Exception:
        print(f"ERROR: {title}\n{message}")


def main():
    """Launch the Personal LLM Desktop Application."""
    print()
    print("=" * 55)
    print("  🧠 Personal LLM — Desktop Application")
    print("  Runs 100% on YOUR hardware. Zero cloud.")
    print("=" * 55)

    # ─── Step 1: Check dependencies ───────────────────
    print("\n🔍 Checking dependencies...")
    try:
        import gradio
    except ImportError as e:
        import traceback
        error_msg = f"Gradio import failed:\n{e}\n\n{traceback.format_exc()}"
        print(error_msg)
        show_error_dialog("Missing Dependencies", error_msg)
        sys.exit(1)

    try:
        import webview
    except ImportError:
        show_error_dialog(
            "Missing Dependencies",
            "pywebview is not installed.\n\n"
            "Run: pip install pywebview"
        )
        sys.exit(1)

    print("✅ Dependencies OK")

    # ─── Step 2: Check for models ─────────────────────
    print("🔍 Checking for models...")
    if not check_models_exist():
        print("⚠️ No models found. Proceeding to Model Manager...")
    else:
        print("✅ Models found")

    # ─── Step 3: Find a free port ─────────────────────
    port = find_free_port()
    host = "127.0.0.1"
    print(f"🔌 Using port {port}")

    # Override the config port so Gradio uses our dynamic port
    from personal_llm import config
    config.UI_PORT = port

    # ─── Step 4: Create splash window ─────────────────
    splash_html = f"""
    <!DOCTYPE html>
    <html>
    <head><style>
        body {{
            margin: 0; display: flex; align-items: center; justify-content: center;
            height: 100vh; font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
        }}
        .container {{ text-align: center; }}
        h1 {{
            font-size: 2.5em; margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        p {{ color: #8b8fa3; font-size: 1.1em; }}
        .spinner {{
            width: 40px; height: 40px; margin: 30px auto;
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #667eea; border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style></head>
    <body>
        <div class="container">
            <h1>🧠 Personal LLM</h1>
            <p>Starting your private AI assistant...</p>
            <div class="spinner"></div>
            <p style="font-size:0.85em; margin-top:20px;">Loading models and initializing engine</p>
        </div>
    </body>
    </html>
    """

    # ─── Step 5: Start FastAPI server in background ────
    server_ready = threading.Event()
    server_error = [None]  # mutable container for error

    def start_server():
        """Start the FastAPI server in a background thread."""
        try:
            from personal_llm.api import launch_api
            launch_api(port=port)
            server_ready.set()
        except Exception as e:
            server_error[0] = str(e)
            logger.error(f"Server failed to start: {e}")
            server_ready.set()  # unblock the wait

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # ─── Step 6: Create the desktop window ────────────
    window = webview.create_window(
        title='Personal LLM — Loading...',
        html=splash_html,
        width=1280,
        height=850,
        resizable=True,
        text_select=True,
    )

    def on_window_shown():
        """After the splash window is visible, wait for server and navigate."""
        # Wait for server thread to signal ready
        server_ready.wait(timeout=60)

        if server_error[0]:
            window.evaluate_js(f"""
                document.body.innerHTML = '<div style="text-align:center;padding:100px;font-family:Segoe UI;color:white;background:#1a1a2e;height:100vh;margin:0;">' +
                '<h1 style="color:#ff6b6b;">❌ Startup Error</h1>' +
                '<p>{server_error[0]}</p>' +
                '<p style="color:#888;">Please check the console for details.</p></div>';
            """)
            return

        # Poll until the server is actually accepting connections
        print("⏳ Waiting for API server to be ready...")
        if wait_for_server(host, port, timeout=45):
            local_url = f"http://{host}:{port}/docs"
            print(f"✅ API ready at {local_url}")
            time.sleep(0.5)  # Brief pause for FastAPI to fully initialize
            window.load_url(local_url)
            window.set_title("Personal LLM API — Developer Docs")
        else:
            window.evaluate_js("""
                document.body.innerHTML = '<div style="text-align:center;padding:100px;font-family:Segoe UI;color:white;background:#1a1a2e;height:100vh;margin:0;">' +
                '<h1 style="color:#ff6b6b;">⏱️ Timeout</h1>' +
                '<p>The API Server took too long to start.</p>' +
                '<p style="color:#888;">Try restarting the application.</p></div>';
            """)

    def on_closing():
        """Clean shutdown when the window is closed."""
        print("🔄 Shutting down...")
        try:
            # Try to unload the LLM engine to free VRAM/RAM
            from personal_llm.llm_engine import get_engine
            engine = get_engine()
            if engine.is_loaded:
                engine.unload()
                print("✅ Model unloaded, VRAM freed.")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    # Bind events
    window.events.shown += on_window_shown
    window.events.closing += on_closing

    # ─── Step 7: Start the GUI event loop ─────────────
    print("🖥️  Opening desktop window...")
    webview.start(gui='edgechromium')  # Use Edge WebView2 for modern CSS/JS support

    # When webview.start() returns, the window has been closed
    print("👋 Goodbye!")
    os._exit(0)


if __name__ == '__main__':
    main()
