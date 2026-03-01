/* eslint-disable */
/**
 * Personal LLM - Electron Main Process
 * Spawns the Python FastAPI backend, then opens the React frontend.
 *
 * Python resolution order:
 * 1. Bundled python-embed/ (if present and has all deps)
 * 2. System Python (python / python3 / py)
 *
 * On first run, auto-installs missing pip packages.
 */
const { app, BrowserWindow, shell, dialog } = require("electron");
const { spawn, execSync, execFile } = require("child_process");
const path = require("path");
const http = require("http");
const fs = require("fs");

const API_PORT = 8000;
const API_URL = `http://127.0.0.1:${API_PORT}`;
const IS_DEV = !app.isPackaged;

const serve = require("electron-serve").default || require("electron-serve");
const loadUI = serve({ directory: path.join(__dirname, "..", "out") });

let pythonProcess = null;
let mainWindow = null;
let splashWindow = null;

// ---- Find Python ----
function findPython() {
  if (!IS_DEV) {
    // Check for bundled embedded Python first
    const embedded = path.join(process.resourcesPath, "python-embed", "python.exe");
    if (fs.existsSync(embedded)) {
      console.log("Found bundled Python:", embedded);
      return embedded;
    }
  }
  // Try system Python
  for (const cmd of ["python", "python3", "py"]) {
    try {
      const ver = execSync(`${cmd} --version`, { encoding: "utf-8", timeout: 5000, windowsHide: true }).trim();
      console.log(`Found system ${cmd}: ${ver}`);
      return cmd;
    } catch { }
  }
  return null;
}

// ---- Get the project root for backend files ----
function getProjectRoot() {
  if (IS_DEV) return path.resolve(__dirname, "..", "..");
  return path.resolve(process.resourcesPath);
}

// ---- Check if a Python module is importable ----
function canImport(pythonExe, mod) {
  try {
    execSync(`${pythonExe} -c "import ${mod}"`, { timeout: 10000, windowsHide: true, stdio: "pipe" });
    return true;
  } catch { return false; }
}

// ---- First-run setup: install missing deps ----
function installDependencies(pythonExe) {
  const required = ["fastapi", "uvicorn", "llama_cpp", "huggingface_hub", "pydantic"];
  const pipMap = {
    fastapi: "fastapi",
    uvicorn: "uvicorn[standard]",
    llama_cpp: "llama-cpp-python",
    huggingface_hub: "huggingface-hub",
    pydantic: "pydantic"
  };

  const missing = required.filter(mod => !canImport(pythonExe, mod));
  if (missing.length === 0) {
    console.log("All Python dependencies are installed.");
    return true;
  }

  console.log("Missing packages:", missing.join(", "));
  const pkgsToInstall = missing.map(m => pipMap[m]).join(" ");

  try {
    // Show progress to user via splash
    if (splashWindow) {
      splashWindow.webContents.executeJavaScript(
        `document.getElementById('status').innerText = 'Installing Python packages (first run only)...'`
      ).catch(() => { });
    }

    console.log("Installing:", pkgsToInstall);
    execSync(`${pythonExe} -m pip install ${pkgsToInstall} --no-warn-script-location`, {
      timeout: 300000, // 5 min timeout
      windowsHide: true,
      encoding: "utf-8",
    });
    console.log("Dependencies installed successfully.");
    return true;
  } catch (err) {
    console.error("Failed to install dependencies:", err.message);
    return false;
  }
}

// ---- Splash screen (shown during first-run setup) ----
function createSplash() {
  splashWindow = new BrowserWindow({
    width: 450,
    height: 300,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });

  const html = `
    <html>
    <body style="margin:0;display:flex;align-items:center;justify-content:center;height:100vh;
      background:linear-gradient(135deg,#0B0E14 0%,#1a1040 50%,#0B0E14 100%);
      font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;color:#fff;
      border-radius:16px;overflow:hidden;">
      <div style="text-align:center;padding:40px;">
        <div style="font-size:48px;margin-bottom:16px;">&#129504;</div>
        <h1 style="font-size:22px;font-weight:700;margin:0 0 8px 0;">Personal LLM</h1>
        <p id="status" style="font-size:13px;color:#818cf8;margin:0 0 24px 0;">Starting AI engine...</p>
        <div style="width:240px;height:6px;background:rgba(255,255,255,0.08);border-radius:100px;margin:0 auto;overflow:hidden;position:relative;">
          <div style="position:absolute;top:0;left:0;height:100%;width:100%;border-radius:100px;
            background:linear-gradient(90deg,transparent 0%,#818cf8 40%,#a78bfa 60%,transparent 100%);
            animation:shimmer 1.8s ease-in-out infinite;"></div>
        </div>
      </div>
    </body>
    <style>
      @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
      }
    </style>
    </html>`;

  splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
  return splashWindow;
}

// ---- Start the Python backend ----
function startPythonBackend(pythonExe) {
  const root = getProjectRoot();
  console.log(`Starting backend: ${pythonExe} -m personal_llm.api (cwd: ${root})`);

  // Embedded Python ignores PYTHONPATH when a ._pth file exists.
  // We must inject the project root into sys.path explicitly via -c.
  const bootScript = `import sys; sys.path.insert(0, r'${root.replace(/\\/g, "\\\\")}'); from personal_llm.api import launch_api; launch_api()`;

  pythonProcess = spawn(pythonExe, ["-c", bootScript], {
    cwd: root,
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONIOENCODING: "utf-8",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout.on("data", (d) => console.log("[Py]", d.toString().trim()));
  pythonProcess.stderr.on("data", (d) => {
    const m = d.toString().trim();
    if (m.includes("ERROR") || m.includes("Traceback")) console.error("[Py ERR]", m);
    else console.log("[Py]", m);
  });
  pythonProcess.on("error", (e) => console.error("[Spawn Error]", e.message));
  pythonProcess.on("close", (c) => { console.log("[Py] exit", c); pythonProcess = null; });
}

// ---- Wait for backend API ----
function waitForAPI(maxRetries = 60) {
  return new Promise((resolve, reject) => {
    let n = 0;
    const iv = setInterval(() => {
      n++;
      if (splashWindow && n % 4 === 0) {
        splashWindow.webContents.executeJavaScript(
          `document.getElementById('status').innerText = 'Starting AI engine... (${n}s)'`
        ).catch(() => { });
      }
      const req = http.get(`${API_URL}/api/status`, (res) => {
        if (res.statusCode === 200) { clearInterval(iv); resolve(); }
      });
      req.on("error", () => { if (n >= maxRetries) { clearInterval(iv); reject(new Error("Timeout")); } });
      req.end();
    }, 500);
  });
}

// ---- Main window ----
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 800, minHeight: 600,
    title: "Personal LLM", backgroundColor: "#0B0E14", autoHideMenuBar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  loadUI(mainWindow);
  mainWindow.on("closed", () => { mainWindow = null; });
}

// ---- App lifecycle ----
app.whenReady().then(async () => {
  // Show splash
  createSplash();

  // Find Python
  const pythonExe = findPython();
  if (!pythonExe) {
    if (splashWindow) splashWindow.close();
    const r = await dialog.showMessageBox({
      type: "error",
      title: "Python Not Found",
      message: "Personal LLM requires Python to run.",
      detail: "Python was not found on this computer.\n\nPlease install Python 3.10+ from python.org and check 'Add Python to PATH' during installation.\n\nAfter installing Python, restart Personal LLM.",
      buttons: ["Download Python", "Quit"],
    });
    if (r.response === 0) shell.openExternal("https://www.python.org/downloads/");
    app.quit();
    return;
  }

  // First-run: install missing packages
  if (splashWindow) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status').innerText = 'Checking Python packages...'`
    ).catch(() => { });
  }
  installDependencies(pythonExe);

  // Start backend
  if (splashWindow) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status').innerText = 'Starting AI engine...'`
    ).catch(() => { });
  }
  startPythonBackend(pythonExe);

  try {
    await waitForAPI();
    console.log("API is ready!");
  } catch (err) {
    console.error("Backend did not start:", err.message);
  }

  // Close splash and show main window
  if (splashWindow) { splashWindow.close(); splashWindow = null; }
  createWindow();

  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on("window-all-closed", () => {
  if (pythonProcess) { pythonProcess.kill(); pythonProcess = null; }
  if (process.platform !== "darwin") app.quit();
});
app.on("before-quit", () => { if (pythonProcess) { pythonProcess.kill(); pythonProcess = null; } });
