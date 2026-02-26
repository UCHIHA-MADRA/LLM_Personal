/* eslint-disable */
/**
 * 🧠 Personal LLM — Electron Main Process
 * Spawns the Python FastAPI backend as a child process,
 * then opens the React frontend in a native desktop window.
 */
const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const API_PORT = 8000;
const API_URL = `http://127.0.0.1:${API_PORT}`;
const IS_DEV = !app.isPackaged;

const serve = require("electron-serve").default || require("electron-serve");
const loadUI = serve({ directory: path.join(__dirname, "..", "out") });

let pythonProcess = null;
let mainWindow = null;

// ─── Spawn the Python FastAPI Backend ────────────────────────
function startPythonBackend() {
  const projectRoot = IS_DEV
    ? path.resolve(__dirname, "..", "..")       // dev: ui/electron/../../ = project root
    : path.resolve(process.resourcesPath);     // prod: resources/

  const pythonExe = IS_DEV ? "python" : path.join(projectRoot, "python", "python.exe");

  console.log(`🐍 Starting Python backend: ${pythonExe} -m personal_llm.api (cwd: ${projectRoot})`);

  pythonProcess = spawn(pythonExe, ["-m", "personal_llm.api"], {
    cwd: projectRoot,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", PYTHONIOENCODING: "utf-8" },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[Python ERR] ${data.toString().trim()}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`[Python] exited with code ${code}`);
    pythonProcess = null;
  });
}

// ─── Wait for the API to become available ────────────────────
function waitForAPI(maxRetries = 60) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const interval = setInterval(() => {
      attempts++;
      const req = http.get(`${API_URL}/api/status`, (res) => {
        if (res.statusCode === 200) {
          clearInterval(interval);
          resolve();
        }
      });
      req.on("error", () => {
        if (attempts >= maxRetries) {
          clearInterval(interval);
          reject(new Error("Backend timed out"));
        }
      });
      req.end();
    }, 500);
  });
}

// ─── Create the Desktop Window ───────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: "Personal LLM",
    backgroundColor: "#0B0E14",
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    // Frameless with custom titlebar for premium feel
    // titleBarStyle: "hidden",
    // titleBarOverlay: { color: "#0B0E14", symbolColor: "#F8FAFC", height: 40 },
  });

  // Open external links in the system browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  console.log(`📄 Loading UI from built export folder`);
  loadUI(mainWindow);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ─── App Lifecycle ───────────────────────────────────────────
app.whenReady().then(async () => {
  // Show a splash-like loading state
  startPythonBackend();

  try {
    console.log("⏳ Waiting for Python API to start...");
    await waitForAPI();
    console.log("✅ API is ready!");
  } catch (err) {
    console.error("❌ Backend failed to start:", err.message);
    // Continue anyway — the UI will show a connection error
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  // Kill the Python backend when the window closes
  if (pythonProcess) {
    console.log("🔄 Shutting down Python backend...");
    pythonProcess.kill();
    pythonProcess = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
});
