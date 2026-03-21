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
const { app, BrowserWindow, shell, dialog, Tray, Menu, nativeImage } = require("electron");
const { spawn, execSync, execFile } = require("child_process");
const path = require("path");
const http = require("http");
const fs = require("fs");

const DEFAULT_API_PORT = 8000;
let apiPort = DEFAULT_API_PORT;
let API_URL = `http://127.0.0.1:${apiPort}`;
const IS_DEV = !app.isPackaged;

const serve = require("electron-serve").default || require("electron-serve");
const loadUI = serve({ directory: path.join(__dirname, "..", "out") });

let pythonProcess = null;
let mainWindow = null;
let splashWindow = null;
let tray = null;
let isQuitting = false; // True when the user explicitly quits (not just closing the window)

// ---- Find Python ----
function findPython() {
  if (!IS_DEV) {
    // In production, the bundled python-embed in resources/ is corrupted by
    // Electron's code signing. Copy it to %LOCALAPPDATA% on first run and use it from there.
    const appData = process.env.LOCALAPPDATA || path.join(require("os").homedir(), "AppData", "Local");
    const localPythonDir = path.join(appData, "PersonalLLM", "python-embed");
    const localPython = path.join(localPythonDir, "python.exe");

    if (fs.existsSync(localPython)) {
      console.log("Found local Python:", localPython);
      return localPython;
    }

    // Copy bundled python-embed to local AppData (first run only)
    const bundledPython = path.join(process.resourcesPath, "python-embed");
    if (fs.existsSync(path.join(bundledPython, "python.exe"))) {
      console.log("First run: Copying Python to", localPythonDir);
      if (splashWindow) {
        splashWindow.webContents.executeJavaScript(
          `document.getElementById('status').innerText = 'First run setup: Extracting Python...'`
        ).catch(() => { });
      }
      try {
        copyDirSync(bundledPython, localPythonDir);
        console.log("Python copied successfully to", localPythonDir);
        return localPython;
      } catch (err) {
        console.error("Failed to copy Python:", err.message);
      }
    }
  }
  // Try system Python as fallback
  for (const cmd of ["python", "python3", "py"]) {
    try {
      const ver = execSync(`${cmd} --version`, { encoding: "utf-8", timeout: 5000, windowsHide: true }).trim();
      console.log(`Found system ${cmd}: ${ver}`);
      return cmd;
    } catch { }
  }
  return null;
}

// ---- Recursive directory copy ----
function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
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
  const required = ["fastapi", "uvicorn", "llama_cpp", "huggingface_hub", "pydantic", "multipart"];
  const pipMap = {
    fastapi: "fastapi",
    uvicorn: "uvicorn[standard]",
    llama_cpp: "llama-cpp-python",
    huggingface_hub: "huggingface-hub",
    pydantic: "pydantic",
    multipart: "python-multipart"
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
    execSync(`${pythonExe} -m pip install ${pkgsToInstall} --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --prefer-binary --no-warn-script-location`, {
      timeout: 300000, // 5 min timeout
      windowsHide: true,
      encoding: "utf-8",
    });
    console.log("Dependencies installed successfully.");
    return true;
  } catch (err) {
    console.error("Failed to install dependencies:", err.message);
    dialog.showMessageBoxSync({
      type: "error",
      title: "Dependency Install Failed",
      message: "Could not install required Python packages.",
      detail: `Ensure you have an internet connection and 'pip' is installed.\n\nError: ${err.message}`,
      buttons: ["OK"]
    });
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
    <head>
      <style>
        body { margin:0; display:flex; align-items:center; justify-content:center; height:100vh; background: #0B0E14; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; color:#ffffff; overflow:hidden; border-radius:16px; user-select:none; }
        .bg-glow { position:absolute; width:300px; height:300px; background:radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%); top:50%; left:50%; transform:translate(-50%,-50%); border-radius:50%; animation: pulse 4s ease-in-out infinite alternate; pointer-events:none; }
        .container { position:relative; z-index:10; text-align:center; padding:40px; width: 100%; box-sizing: border-box; }
        .icon-box { width:72px; height:72px; margin:0 auto 24px; background:linear-gradient(135deg, #4f46e5 0%, #8b5cf6 100%); border-radius:20px; display:flex; items-center; justify-content:center; box-shadow:0 10px 25px rgba(79,70,229,0.4); animation: float 6s ease-in-out infinite; }
        .icon { font-size:36px; line-height:72px; }
        h1 { font-size:24px; font-weight:800; margin:0 0 8px 0; letter-spacing:-0.5px; }
        .version { font-size:11px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom: 24px; display:block; }
        .status-container { display:flex; justify-content:space-between; margin-bottom:8px; width:260px; margin-left:auto; margin-right:auto; }
        #status { font-size:12px; color:#9ca3af; font-weight:500; margin:0; }
        #percent { font-size:12px; color:#818cf8; font-weight:600; margin:0; }
        .progress-track { width:260px; height:4px; background:rgba(255,255,255,0.05); border-radius:100px; margin:0 auto; overflow:hidden; position:relative; border: 1px solid rgba(255,255,255,0.05); }
        .progress-fill { position:absolute; top:0; left:0; height:100%; width:0%; border-radius:100px; background:linear-gradient(90deg, #6366f1 0%, #a855f7 100%); transition: width 0.2s ease-out; box-shadow: 0 0 10px rgba(99,102,241,0.5); }
        .progress-shimmer { position:absolute; top:0; left:0; height:100%; width:100%; border-radius:100px; background:linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%); animation: shimmer 1.5s ease-in-out infinite; transform:translateX(-100%); }
        @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-8px); } 100% { transform: translateY(0px); } }
        @keyframes pulse { 0% { opacity: 0.5; transform: translate(-50%,-50%) scale(0.9); } 100% { opacity: 1; transform: translate(-50%,-50%) scale(1.1); } }
        @keyframes shimmer { 100% { transform: translateX(100%); } }
      </style>
    </head>
    <body>
      <div class="bg-glow"></div>
      <div class="container">
        <div class="icon-box"><div class="icon">&#129504;</div></div>
        <h1>Personal LLM</h1>
        <span class="version">Version 2.0.0</span>
        
        <div class="status-container">
          <p id="status">Waking up engine...</p>
          <p id="percent">0%</p>
        </div>
        
        <div class="progress-track">
          <div class="progress-fill" id="fill"><div class="progress-shimmer"></div></div>
        </div>
      </div>
      
      <script>
        const statusEl = document.getElementById('status');
        const percentEl = document.getElementById('percent');
        const fillEl = document.getElementById('fill');
        
        const phases = [
          {p: 5, t: "Initializing Python runtime..."},
          {p: 15, t: "Loading FastAPI backend..."},
          {p: 35, t: "Mounting SQLite context engine..."},
          {p: 60, t: "Warming up local AI node..."},
          {p: 85, t: "Connecting to UI renderer..."},
          {p: 95, t: "Almost ready..."}
        ];
        
        let currentP = 0;
        let phaseIdx = 0;
        
        function updateProgress() {
          if (phaseIdx < phases.length) {
            const target = phases[phaseIdx];
            if (currentP < target.p) {
              currentP += (Math.random() * 2);
              if (currentP > target.p) currentP = target.p;
              
              fillEl.style.width = currentP + '%';
              percentEl.innerText = Math.floor(currentP) + '%';
              statusEl.innerText = target.t;
            } else {
              phaseIdx++;
            }
          }
          requestAnimationFrame(updateProgress);
        }
        updateProgress();
      </script>
    </body>
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
      RESOURCES_PATH: process.resourcesPath || "",
      ELECTRON_MODE: "1",  // Tells api.py to bind to 127.0.0.1 (no firewall needed)
      PERSONAL_LLM_DIR: !IS_DEV ? path.join(app.getPath("userData"), "PersonalLLM") : "",
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
function waitForAPI(maxRetries = 120) {
  // Read the port file that Python writes after finding an available port
  const portFile = path.join(require("os").homedir(), ".personal_llm_port");

  return new Promise((resolve, reject) => {
    let n = 0;
    const iv = setInterval(() => {
      n++;

      // Check if Python has written its port file
      if (n % 2 === 0) {
        try {
          if (fs.existsSync(portFile)) {
            const port = parseInt(fs.readFileSync(portFile, "utf-8").trim(), 10);
            if (port && port !== apiPort) {
              apiPort = port;
              API_URL = `http://127.0.0.1:${apiPort}`;
              console.log(`[Main] Detected backend on port ${apiPort}`);
            }
          }
        } catch { }
      }

      if (splashWindow && n % 5 === 0) {
        splashWindow.webContents.executeJavaScript(
          `document.getElementById('status').innerText = 'Starting AI engine... (${n}s)'`
        ).catch(() => { });
      }
      const req = http.get(`${API_URL}/api/status`, (res) => {
        if (res.statusCode === 200) { clearInterval(iv); resolve(); }
      });
      req.on("error", () => { if (n >= maxRetries) { clearInterval(iv); reject(new Error("Timeout")); } });
      req.end();
    }, 1000);
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

  // Minimize to tray instead of closing
  mainWindow.on("close", (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
  mainWindow.on("closed", () => { mainWindow = null; });
}

// ---- System Tray ----
function createTray() {
  // Use favicon.ico as the tray icon
  const iconPath = path.join(__dirname, "..", "app", "favicon.ico");
  let trayIcon;
  if (fs.existsSync(iconPath)) {
    trayIcon = nativeImage.createFromPath(iconPath);
  } else {
    // Fallback: create a simple 16x16 icon
    trayIcon = nativeImage.createEmpty();
  }

  tray = new Tray(trayIcon);
  tray.setToolTip("Personal LLM — 100% Local AI");

  const contextMenu = Menu.buildFromTemplate([
    { label: "Show Personal LLM", click: () => { if (mainWindow) { mainWindow.show(); mainWindow.focus(); } } },
    { type: "separator" },
    { label: "Quit", click: () => { isQuitting = true; app.quit(); } },
  ]);
  tray.setContextMenu(contextMenu);
  tray.on("double-click", () => { if (mainWindow) { mainWindow.show(); mainWindow.focus(); } });
}

// ---- Graceful Shutdown: unload model before killing Python ----
function gracefulShutdown() {
  return new Promise((resolve) => {
    if (!pythonProcess) { resolve(); return; }

    // Try to unload model via API first
    const req = http.request(`${API_URL}/api/models/unload`, { method: "POST", timeout: 3000 }, (res) => {
      console.log("[Shutdown] Model unloaded, status:", res.statusCode);
      resolve();
    });
    req.on("error", () => { resolve(); }); // API might already be down
    req.on("timeout", () => { req.destroy(); resolve(); });
    req.end();

    // Hard deadline: 4 seconds max wait
    setTimeout(resolve, 4000);
  }).then(() => {
    if (pythonProcess) {
      pythonProcess.kill();
      pythonProcess = null;
    }
  });
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

  // Capture Python stderr for diagnostics
  let pythonErrors = [];
  startPythonBackend(pythonExe);
  if (pythonProcess) {
    pythonProcess.stderr.removeAllListeners("data");
    pythonProcess.stderr.on("data", (d) => {
      const m = d.toString().trim();
      pythonErrors.push(m);
      if (m.includes("ERROR") || m.includes("Traceback")) console.error("[Py ERR]", m);
      else console.log("[Py]", m);
    });
  }

  try {
    await waitForAPI();
    console.log("API is ready!");

    // Close splash and show main window
    if (splashWindow) { splashWindow.close(); splashWindow = null; }
    createTray();
    createWindow();
  } catch (err) {
    console.error("Backend did not start:", err.message);

    // Show diagnostic error to user instead of hanging
    if (splashWindow) { splashWindow.close(); splashWindow = null; }
    const errorLog = pythonErrors.join("\n");
    const pythonAlive = pythonProcess !== null;

    // Check for missing C++ Redistributable (classic WinError 1114 / 126 from llama.dll)
    if (errorLog.includes("WinError 1114") || errorLog.includes("WinError 126") || errorLog.includes("VCRUNTIME140.dll")) {
      const redistPath = path.join(process.resourcesPath, "vc_redist.x64.exe");

      if (fs.existsSync(redistPath)) {
        dialog.showMessageBoxSync({
          type: "warning",
          title: "Installing Required Components",
          message: "Your PC is missing the Microsoft Visual C++ Redistributable.",
          detail: "The AI engine requires this official Microsoft component.\n\nClick OK to install it automatically. Administrator permission may be required.",
          buttons: ["OK"],
        });

        try {
          // Run the included redist installer with admin elevation (triggers UAC prompt)
          execSync(`powershell -Command "Start-Process -FilePath '${redistPath}' -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait"`, {
            timeout: 120000,
            windowsHide: true,
          });

          dialog.showMessageBoxSync({
            type: "info",
            title: "Installation Complete",
            message: "Success! The required components have been installed.",
            detail: "Personal LLM will now restart.",
            buttons: ["Restart"]
          });

          app.relaunch();
        } catch (installErr) {
          dialog.showMessageBoxSync({
            type: "error",
            title: "Installation Failed",
            message: "Failed to automatically install the C++ Redistributable.",
            detail: `Please run it manually from:\n${redistPath}\n\nError: ${installErr.message}`,
            buttons: ["Quit"]
          });
        }
      } else {
        // Fallback if the redist isn't bundled for some reason
        const r = dialog.showMessageBoxSync({
          type: "error",
          title: "Missing System Components",
          message: "Your PC is missing the Microsoft Visual C++ Redistributable.",
          detail: "The AI engine (llama-cpp-python) requires this official Microsoft component to run on Windows.\n\nPlease download and install 'vc_redist.x64.exe', then restart Personal LLM.",
          buttons: ["Download vc_redist", "Quit"],
        });
        if (r === 0) shell.openExternal("https://aka.ms/vs/17/release/vc_redist.x64.exe");
      }
    } else {
      // Generic crash
      dialog.showMessageBoxSync({
        type: "error",
        title: "Backend Failed to Start",
        message: "The AI engine could not start.",
        detail: `Python: ${pythonExe}\nProcess alive: ${pythonAlive}\n\n--- Last Python Output ---\n${errorLog.slice(-1500) || "(no output captured)"}\n\nTry:\n1. Run the installer as Administrator\n2. Check if antivirus is blocking python.exe`,
        buttons: ["Quit"],
      });
    }

    app.quit();
  }

  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on("window-all-closed", () => {
  // On macOS, apps stay open until explicitly quit
  if (process.platform !== "darwin" && isQuitting) app.quit();
});

app.on("before-quit", async (e) => {
  if (pythonProcess) {
    e.preventDefault();
    console.log("[Shutdown] Gracefully shutting down...");
    await gracefulShutdown();
    if (tray) { tray.destroy(); tray = null; }
    app.exit(0);
  }
});
