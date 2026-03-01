"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Bot, Send, User, Menu, Settings, X, Trash2, Key, Cloud,
  Cpu, Zap, ChevronDown, Package, PlusCircle, AlertCircle, CheckCircle2, Loader2
} from "lucide-react"
import { cn } from "./lib/utils"

// ─── TYPES ───
type Role = "user" | "assistant" | "system"
type Message = {
  id: string
  role: Role
  content: string
}
type ModelStatus = {
  loaded: boolean
  name?: string
  size_gb?: number
  context_window?: number
}
type ConversationItem = { id: string; title: string; updated_at: string; model?: string; message_count?: number }
type CatalogEntry = {
  key: string; name: string; description: string; size_gb: number;
  filename: string; is_downloaded: boolean; fit_info: Record<string, unknown>
}
type Toast = { id: string; message: string; type: "success" | "error" | "info" }
type DownloadStatus = { active: boolean; key: string; progress: number; message: string; done: boolean; error: string | null }

// ─── TOAST COMPONENT ───
function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed bottom-6 right-6 z-100 flex flex-col gap-3 max-w-sm">
      <AnimatePresence>
        {toasts.map(t => (
          <motion.div key={t.id} initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 50 }}
            className={cn(
              "p-4 rounded-xl border backdrop-blur-lg shadow-2xl flex items-start gap-3 text-sm cursor-pointer",
              t.type === "error" ? "bg-red-500/10 border-red-500/20 text-red-300" :
                t.type === "success" ? "bg-green-500/10 border-green-500/20 text-green-300" :
                  "bg-indigo-500/10 border-indigo-500/20 text-indigo-300"
            )}
            onClick={() => onDismiss(t.id)}
          >
            {t.type === "error" ? <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" /> :
              t.type === "success" ? <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" /> :
                <Loader2 className="w-5 h-5 shrink-0 mt-0.5 animate-spin" />}
            <span>{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

// ─── MAIN COMPONENT ───
export default function PersonalLLMApp() {
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "assistant", content: "Hello! I am your Personal LLM. How can I help you today?" }
  ])
  const [input, setInput] = useState("")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isGenerating, setIsGenerating] = useState(false)
  const [status, setStatus] = useState<ModelStatus>({ loaded: false })
  const endOfMessagesRef = useRef<HTMLDivElement>(null)

  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<"models" | "apikeys">("models")
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [backendConnected, setBackendConnected] = useState(false)

  // API Key states
  const [apiKeys, setApiKeys] = useState({ openai_key: "", groq_key: "", together_key: "" })
  const [savedKeysMasked, setSavedKeysMasked] = useState<Record<string, string>>({})

  // Toast helpers
  const addToast = useCallback((message: string, type: Toast["type"] = "info") => {
    const id = Date.now().toString()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
  }, [])

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  // ─── Data Fetching ───
  const fetchStatus = useCallback(() => {
    fetch("http://localhost:8000/api/status")
      .then(res => res.json())
      .then(data => { setStatus(data); setBackendConnected(true) })
      .catch(() => { setBackendConnected(false) })
  }, [])

  const fetchData = useCallback(() => {
    fetch("http://localhost:8000/api/conversations")
      .then(res => res.json())
      .then(data => setConversations(data.sort((a: ConversationItem, b: ConversationItem) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())))
      .catch(() => { })

    fetch("http://localhost:8000/api/models")
      .then(res => res.json())
      .then(data => setCatalog(data.catalog))
      .catch(() => { })
  }, [])

  const fetchSettings = useCallback(() => {
    fetch("http://localhost:8000/api/settings")
      .then(res => res.json())
      .then(data => setSavedKeysMasked(data))
      .catch(() => { })
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchData()
    fetchSettings()
  }, [fetchStatus, fetchData, fetchSettings])

  // Auto-retry backend connection every 3s when disconnected
  useEffect(() => {
    if (backendConnected) return
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/status")
        .then(res => res.json())
        .then(data => {
          setStatus(data)
          setBackendConnected(true)
          fetchData()
          fetchSettings()
          addToast("Backend connected!", "success")
        })
        .catch(() => { })
    }, 3000)
    return () => clearInterval(interval)
  }, [backendConnected, fetchData, fetchSettings, addToast])

  // Auto-scroll chat
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // ─── Download Progress Polling ───
  useEffect(() => {
    if (!downloadStatus?.active) return
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/models/download/status")
        .then(res => res.json())
        .then((data: DownloadStatus) => {
          setDownloadStatus(data)
          if (data.done) {
            clearInterval(interval)
            if (data.error) {
              addToast(data.message || "Download failed", "error")
            } else {
              addToast(data.message || "Download complete!", "success")
            }
            fetchData() // Refresh model list
            setDownloadStatus(null)
          }
        })
        .catch(() => { })
    }, 2000)
    return () => clearInterval(interval)
  }, [downloadStatus?.active, addToast, fetchData])

  // ─── Conversation Actions ───
  const loadConversation = async (id: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/conversations/${id}`)
      if (res.ok) {
        const data = await res.json()
        setCurrentConversationId(data.id)
        setMessages(data.messages)
        if (window.innerWidth < 768) setIsSidebarOpen(false)
      }
    } catch { addToast("Failed to load conversation", "error") }
  }

  const deleteConversation = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await fetch(`http://localhost:8000/api/conversations/${id}`, { method: "DELETE" })
      if (currentConversationId === id) newConversation()
      fetchData()
      addToast("Conversation deleted", "success")
    } catch { addToast("Failed to delete conversation", "error") }
  }

  const newConversation = () => {
    if (!currentConversationId && messages.length <= 1) {
      addToast("Already on a new conversation", "info")
      if (window.innerWidth < 768) setIsSidebarOpen(false)
      return
    }
    setCurrentConversationId(null)
    setMessages([{ id: "1", role: "assistant", content: "Hello! I am your Personal LLM. How can I help you today?" }])
    addToast("Started new conversation", "success")
    if (window.innerWidth < 768) setIsSidebarOpen(false)
  }

  // ─── Chat ───
  const handleSend = async () => {
    if (!input.trim() || isGenerating) return
    if (!status.loaded) {
      addToast("No model loaded. Open Settings to load one.", "error")
      return
    }

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: input }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setIsGenerating(true)

    try {
      const asstId = (Date.now() + 1).toString()
      setMessages(prev => [...prev, { id: asstId, role: "assistant", content: "" }])

      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          conversation_id: currentConversationId,
          max_tokens: 1024,
          temperature: 0.7
        })
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Unknown error" }))
        throw new Error(err.detail || "Stream failed")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      while (reader) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'init') {
                setCurrentConversationId(data.conversation_id)
              } else if (data.type === 'token') {
                setMessages(prev => prev.map(msg =>
                  msg.id === asstId ? { ...msg, content: msg.content + data.content } : msg
                ))
              } else if (data.type === 'error') {
                addToast(data.content || "Generation error", "error")
                break
              } else if (data.type === 'done') {
                break
              }
            } catch { }
          }
        }
      }
      fetchData()
    } catch (e) {
      addToast(e instanceof Error ? e.message : "Chat failed", "error")
    } finally {
      setIsGenerating(false)
    }
  }

  // ─── Model Load ───
  const handleLoadModel = async (filename: string) => {
    setIsLoadingModel(true)
    try {
      const res = await fetch("http://localhost:8000/api/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename })
      })
      if (res.ok) {
        fetchStatus()
        setIsSettingsOpen(false)
        addToast("Model loaded successfully!", "success")
      } else {
        addToast("Failed to load model", "error")
      }
    } catch { addToast("Network error loading model", "error") }
    finally { setIsLoadingModel(false) }
  }

  // ─── Model Download ───
  const handleDownloadModel = async (key: string) => {
    try {
      const res = await fetch("http://localhost:8000/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ catalog_key: key })
      })
      if (res.ok) {
        setDownloadStatus({ active: true, key, progress: 0, message: "Starting...", done: false, error: null })
        addToast("Download started!", "info")
      } else {
        const err = await res.json().catch(() => ({ detail: "Failed" }))
        addToast(err.detail || "Download failed to start", "error")
      }
    } catch { addToast("Network error starting download", "error") }
  }

  // ─── Save API Keys ───
  const handleSaveKeys = async () => {
    try {
      const body: Record<string, string> = {}
      if (apiKeys.openai_key) body.openai_key = apiKeys.openai_key
      if (apiKeys.groq_key) body.groq_key = apiKeys.groq_key
      if (apiKeys.together_key) body.together_key = apiKeys.together_key

      const res = await fetch("http://localhost:8000/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })
      if (res.ok) {
        addToast("API keys saved!", "success")
        setApiKeys({ openai_key: "", groq_key: "", together_key: "" })
        fetchSettings()
      } else {
        addToast("Failed to save keys", "error")
      }
    } catch { addToast("Network error saving keys", "error") }
  }

  return (
    <div className="flex h-screen w-full overflow-hidden bg-linear-to-br from-[#050510] via-[#0f0c29] to-[#302b63]">

      {/* ─── TOAST NOTIFICATIONS ─── */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* ─── SETTINGS MODAL ─── */}
      <AnimatePresence>
        {isSettingsOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsSettingsOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-surface-900 border border-white/10 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-5 border-b border-white/10 flex justify-between items-center">
                <h3 className="text-lg font-semibold text-white">Models & Settings</h3>
                <button onClick={() => setIsSettingsOpen(false)} className="text-gray-400 hover:text-white"><X className="w-5 h-5" /></button>
              </div>

              {/* Tabs */}
              <div className="flex border-b border-white/10">
                <button onClick={() => setSettingsTab("models")} className={cn("flex-1 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2",
                  settingsTab === "models" ? "text-indigo-400 border-b-2 border-indigo-400" : "text-gray-400 hover:text-white"
                )}><Package className="w-4 h-4" /> Models</button>
                <button onClick={() => setSettingsTab("apikeys")} className={cn("flex-1 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2",
                  settingsTab === "apikeys" ? "text-indigo-400 border-b-2 border-indigo-400" : "text-gray-400 hover:text-white"
                )}><Key className="w-4 h-4" /> API Keys</button>
              </div>

              <div className="p-5 overflow-y-auto flex-1">
                {settingsTab === "models" ? (
                  <>
                    {/* Downloaded Section */}
                    <div className="mb-6">
                      <h4 className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4"><Package className="w-4 h-4" /> Downloaded & Ready</h4>
                      {catalog.filter(c => c.is_downloaded).length === 0 ? (
                        <p className="text-sm text-gray-400 p-4 border border-dashed border-white/10 rounded-xl text-center">No models downloaded yet.</p>
                      ) : (
                        <div className="space-y-3">
                          {catalog.filter(c => c.is_downloaded).map(model => (
                            <div key={model.key} className="p-4 rounded-xl border border-indigo-500/20 bg-indigo-500/5 flex items-center justify-between hover:bg-indigo-500/10 transition-colors">
                              <div>
                                <div className="font-medium text-indigo-300 text-sm">{model.name}</div>
                                <div className="text-xs text-indigo-400 mt-1">{(model.fit_info?.fit_level as string) || "Unknown fit"} • {model.size_gb} GB</div>
                              </div>
                              <button disabled={isLoadingModel} onClick={() => handleLoadModel(model.filename)}
                                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm rounded-lg transition-colors shadow-lg shadow-indigo-500/20 shrink-0"
                              >{isLoadingModel ? "Loading..." : "Load Model"}</button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Available for Download */}
                    <div>
                      <h4 className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4"><Zap className="w-4 h-4" /> Available To Download</h4>
                      <div className="space-y-3">
                        {(catalog.filter(c => !c.is_downloaded).length > 0
                          ? catalog.filter(c => !c.is_downloaded)
                          : [
                            { key: "qwen3-1.7b", name: "Qwen3 1.7B (Q4_K_M)", description: "Alibaba's tiny Qwen3. Runs on anything. Great starter model. Apache 2.0.", size_gb: 1.2, filename: "Qwen_Qwen3-1.7B-Q4_K_M.gguf", is_downloaded: false, fit_info: { fit_level: "Perfect" } },
                            { key: "mistral-7b", name: "Mistral 7B Instruct v0.3", description: "Mistral's flagship 7B. Exceptional quality-to-size ratio. Apache 2.0.", size_gb: 4.4, filename: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", is_downloaded: false, fit_info: { fit_level: "Optimal" } },
                            { key: "deepseek-r1-7b", name: "DeepSeek-R1 Qwen Distill 7B", description: "DeepSeek's reasoning model distilled to 7B. Chain-of-thought specialist. MIT license.", size_gb: 4.7, filename: "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", is_downloaded: false, fit_info: { fit_level: "Optimal" } },
                            { key: "qwen3-8b", name: "Qwen3 8B (Q4_K_M)", description: "Alibaba's Qwen3 8B. Strong multilingual + reasoning. Apache 2.0.", size_gb: 5.0, filename: "Qwen_Qwen3-8B-Q4_K_M.gguf", is_downloaded: false, fit_info: { fit_level: "Good" } },
                            { key: "deepseek-coder-6.7b", name: "DeepSeek Coder 6.7B Instruct", description: "DeepSeek's dedicated code model. Excels at programming tasks.", size_gb: 4.0, filename: "deepseek-coder-6.7b-instruct.Q4_K_M.gguf", is_downloaded: false, fit_info: { fit_level: "Optimal" } },
                          ]
                        ).map(model => {
                          const fitLevel = (model.fit_info?.fit_level as string) || "Unknown"
                          const isGoodFit = fitLevel.toLowerCase().includes("optimal") || fitLevel.toLowerCase().includes("perfect")
                          const isPoorFit = fitLevel.toLowerCase().includes("poor") || fitLevel.toLowerCase().includes("unsupported")
                          const isDownloading = downloadStatus?.active && downloadStatus.key === model.key

                          return (
                            <div key={model.key} className="p-4 rounded-xl border border-white/5 bg-white/5 hover:bg-white/10 transition-colors">
                              <div className="flex items-start sm:items-center justify-between flex-col sm:flex-row gap-4">
                                <div className="flex-1">
                                  <div className="font-medium text-white text-sm">{model.name}</div>
                                  <p className="text-xs text-gray-400 mt-1 line-clamp-2 pr-4">{model.description}</p>
                                  <div className="text-xs mt-2 flex items-center gap-2">
                                    <span className="text-gray-500">{model.size_gb} GB</span>
                                    <span className="text-gray-600">•</span>
                                    <span className={cn(
                                      "border px-2 py-0.5 rounded-full",
                                      isGoodFit ? "text-green-400 border-green-400/20 bg-green-400/10"
                                        : isPoorFit ? "text-red-400 border-red-400/20 bg-red-400/10"
                                          : "text-yellow-400 border-yellow-400/20 bg-yellow-400/10"
                                    )}>{fitLevel}</span>
                                  </div>
                                </div>
                                <button
                                  disabled={!!downloadStatus?.active}
                                  onClick={() => handleDownloadModel(model.key)}
                                  className="w-full sm:w-auto px-4 py-2 bg-white/10 hover:bg-white/20 disabled:opacity-50 text-white text-sm rounded-lg transition-colors border border-white/10 shrink-0"
                                >{isDownloading ? "Downloading..." : "Download"}</button>
                              </div>
                              {/* Progress Bar */}
                              {isDownloading && downloadStatus && (
                                <div className="mt-3">
                                  <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                                    <div className="h-full bg-indigo-500 rounded-full transition-all duration-500" style={{ width: `${Math.round(downloadStatus.progress * 100)}%` }} />
                                  </div>
                                  <div className="text-xs text-gray-400 mt-1">{downloadStatus.message} ({Math.round(downloadStatus.progress * 100)}%)</div>
                                </div>
                              )}
                            </div>
                          )
                        })}
                        {catalog.length === 0 && (
                          <div className="mt-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                            <p className="text-xs text-yellow-400">⏳ Showing built-in catalog. The backend is still starting — models will auto-refresh when ready.</p>
                            <button onClick={() => { fetchStatus(); fetchData(); }} className="mt-2 text-xs text-indigo-400 hover:text-indigo-300 underline">Retry Connection</button>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  /* API Keys Tab */
                  <div className="space-y-6">
                    <p className="text-sm text-gray-400">Connect cloud LLM providers to use alongside your local models. Keys are stored locally and never leave your machine.</p>

                    {[
                      { label: "OpenAI", field: "openai_key" as const, placeholder: "sk-...", icon: "🟢" },
                      { label: "Groq", field: "groq_key" as const, placeholder: "gsk_...", icon: "🟠" },
                      { label: "Together AI", field: "together_key" as const, placeholder: "tog_...", icon: "🔵" },
                    ].map(provider => (
                      <div key={provider.field} className="p-4 rounded-xl border border-white/5 bg-white/5">
                        <label className="flex items-center gap-2 text-sm font-medium text-white mb-2">
                          <Cloud className="w-4 h-4 text-indigo-400" />
                          {provider.icon} {provider.label}
                        </label>
                        {savedKeysMasked[provider.field] && (
                          <div className="text-xs text-green-400 mb-2 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Saved: {savedKeysMasked[provider.field]}
                          </div>
                        )}
                        <input
                          type="password"
                          placeholder={provider.placeholder}
                          value={apiKeys[provider.field]}
                          onChange={e => setApiKeys(prev => ({ ...prev, [provider.field]: e.target.value }))}
                          className="w-full bg-surface-900 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none"
                        />
                      </div>
                    ))}

                    <button onClick={handleSaveKeys}
                      className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-medium transition-colors shadow-lg shadow-indigo-500/20"
                    >Save API Keys</button>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* ─── SIDEBAR ─── */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.aside
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            className="w-80 h-full glass-panel flex flex-col border-r border-[#ffffff10] z-20 shrink-0"
          >
            {/* Header */}
            <div className="p-6 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-linear-to-br from-primary-500 to-[#a855f7] flex items-center justify-center shadow-lg shadow-indigo-500/20">
                  <Cpu className="text-white w-5 h-5" />
                </div>
                <div>
                  <h1 className="font-bold text-lg tracking-tight bg-clip-text text-transparent bg-linear-to-r from-white to-gray-400">Personal LLM</h1>
                  <p className="text-xs text-indigo-400 font-medium">100% Local Inference</p>
                </div>
              </div>
              <button onClick={() => setIsSidebarOpen(false)} className="p-2 hover:bg-white/5 rounded-lg transition-colors text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* New Chat Button */}
            <div className="px-4 pb-4 shrink-0">
              <button onClick={newConversation}
                className="w-full py-3 px-4 rounded-xl border border-white/10 hover:border-indigo-500/50 bg-white/5 hover:bg-indigo-500/10 transition-all flex items-center gap-3 text-sm font-medium"
              ><PlusCircle className="w-4 h-4 text-indigo-400" /> New Conversation</button>
            </div>

            {/* Loaded Model Status */}
            <div className="px-4 py-2 shrink-0">
              <div className="p-4 rounded-xl bg-surface-900/80 border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Active Model</span>
                  <div className={cn("w-2 h-2 rounded-full", status.loaded ? "bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]" : "bg-red-500")} />
                </div>
                <div className="font-medium text-sm truncate">{status.loaded ? status.name : "No Model Loaded"}</div>
                {status.loaded && (
                  <div className="flex items-center gap-4 mt-3 text-xs text-gray-400">
                    <span className="flex items-center gap-1.5"><Package className="w-3 h-3" /> {status.size_gb} GB</span>
                    <span className="flex items-center gap-1.5"><Zap className="w-3 h-3" /> {status.context_window} Ctx</span>
                  </div>
                )}
              </div>
            </div>

            {/* Recent Chats */}
            <nav className="flex-1 px-4 py-4 space-y-1 overflow-y-auto min-h-[150px]">
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2">Recent Chats</div>
              {conversations.length === 0 && (
                <div className="text-xs text-gray-500 px-2 italic">No recent chats.</div>
              )}
              {conversations.map((conv) => (
                <div key={conv.id} className="group relative">
                  <button
                    onClick={() => loadConversation(conv.id)}
                    className={cn(
                      "w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors truncate pr-10",
                      currentConversationId === conv.id ? "bg-indigo-500/20 text-indigo-300" : "text-gray-300 hover:bg-white/5 hover:text-white"
                    )}
                  >{conv.title || "Untitled Chat"}</button>
                  <button
                    onClick={(e) => deleteConversation(conv.id, e)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-gray-500 hover:text-red-400 hover:bg-red-400/10 opacity-0 group-hover:opacity-100 transition-all"
                    title="Delete conversation"
                  ><Trash2 className="w-3.5 h-3.5" /></button>
                </div>
              ))}
            </nav>

            {/* Footer Settings */}
            <div className="p-4 border-t border-white/5 shrink-0">
              <button onClick={() => setIsSettingsOpen(true)}
                className="w-full py-2.5 px-3 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors flex items-center gap-3"
              ><Settings className="w-4 h-4" /> Settings & Models</button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ─── MAIN CHAT AREA ─── */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#0B0E14]/80 backdrop-blur-xl">
        {/* Topbar */}
        <header className="h-16 shrink-0 border-b border-white/5 flex items-center px-4 justify-between">
          <div className="flex items-center gap-3">
            {!isSidebarOpen && (
              <button onClick={() => setIsSidebarOpen(true)} className="p-2 hover:bg-white/5 rounded-lg text-gray-400 hover:text-white transition-colors">
                <Menu className="w-5 h-5" />
              </button>
            )}
            <h2 className="font-medium text-white truncate max-w-[200px] sm:max-w-md">
              {conversations.find(c => c.id === currentConversationId)?.title || "New Conversation"}
            </h2>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {!backendConnected && (
              <span className="flex items-center gap-1.5 text-xs text-red-400">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                Connecting...
              </span>
            )}
            <button onClick={() => setIsSettingsOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm transition-colors max-w-[200px]"
            >
              {backendConnected && <span className={cn("w-2 h-2 rounded-full shrink-0", status.loaded ? "bg-green-500" : "bg-yellow-500")} />}
              <span className="truncate">{status.loaded ? status.name : "Select Model"}</span> <ChevronDown className="w-4 h-4 text-gray-400 shrink-0" />
            </button>
          </div>
        </header>

        {/* ─── WELCOME WIZARD (first-run) ─── */}
        {!status.loaded && conversations.length === 0 && messages.length <= 1 && (
          <div className="flex-1 flex items-center justify-center p-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-lg text-center"
            >
              <div className="text-6xl mb-6">🧠</div>
              <h2 className="text-2xl font-bold text-white mb-3">Welcome to Personal LLM</h2>
              <p className="text-gray-400 mb-8 leading-relaxed">
                Your private AI assistant that runs 100% on your hardware.<br />
                No cloud, no data collection, no subscription fees.
              </p>

              <div className="space-y-4 text-left mb-8">
                <div className="flex items-start gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
                  <span className="text-xl">1️⃣</span>
                  <div>
                    <div className="text-sm font-medium text-white">Download an AI Model</div>
                    <div className="text-xs text-gray-400 mt-1">Pick from our curated catalog. We recommend starting with Llama 3.2 3B (~2 GB).</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
                  <span className="text-xl">2️⃣</span>
                  <div>
                    <div className="text-sm font-medium text-white">Load the Model</div>
                    <div className="text-xs text-gray-400 mt-1">Click &quot;Load&quot; to activate it. The model runs entirely on your PC.</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
                  <span className="text-xl">3️⃣</span>
                  <div>
                    <div className="text-sm font-medium text-white">Start Chatting!</div>
                    <div className="text-xs text-gray-400 mt-1">Your conversations are private and stored locally. No internet required.</div>
                  </div>
                </div>
              </div>

              <button
                onClick={() => { setIsSettingsOpen(true); setSettingsTab("models"); }}
                className="px-8 py-3.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold text-sm transition-all shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 hover:-translate-y-0.5"
              >
                <Package className="w-4 h-4 inline mr-2" />
                Browse & Download Models
              </button>
              <p className="text-xs text-gray-500 mt-4">Free. No account needed. Models from HuggingFace.</p>
            </motion.div>
          </div>
        )}

        {/* Chat History */}
        {(status.loaded || conversations.length > 0 || messages.length > 1) && (
          <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth">
            <div className="max-w-3xl mx-auto space-y-8">
              {messages.map((msg, idx) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={cn("flex gap-4", msg.role === "user" ? "flex-row-reverse" : "flex-row")}
                >
                  <div className={cn(
                    "w-10 h-10 shrink-0 rounded-xl flex items-center justify-center shadow-lg",
                    msg.role === "user"
                      ? "bg-linear-to-br from-indigo-500 to-purple-600 shadow-indigo-500/20"
                      : "bg-[#1E2330] border border-white/10"
                  )}>
                    {msg.role === "user" ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-indigo-400" />}
                  </div>
                  <div className={cn(
                    "max-w-[80%] rounded-2xl p-4 leading-relaxed text-sm whitespace-pre-wrap",
                    msg.role === "user"
                      ? "bg-indigo-600 text-white rounded-tr-none shadow-xl shadow-indigo-900/20"
                      : "glass-panel text-gray-200 rounded-tl-none shadow-xl shadow-black/20"
                  )}>
                    {msg.content || (isGenerating && idx === messages.length - 1 ? (
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse delay-75" />
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse delay-150" />
                      </span>
                    ) : "")}
                  </div>
                </motion.div>
              ))}
              <div ref={endOfMessagesRef} />
            </div>
          </div>
        )}

        {/* Input Box */}
        <div className="p-4 md:p-6 bg-linear-to-t from-[#0B0E14] to-transparent">
          <div className="max-w-3xl mx-auto relative group">
            <div className="absolute -inset-1 rounded-2xl bg-linear-to-r from-indigo-500 to-purple-500 opacity-20 group-focus-within:opacity-50 blur transition duration-500"></div>
            <div className="relative flex items-center bg-surface-900 rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Message your Local LLM..."
                className="flex-1 max-h-48 min-h-[60px] bg-transparent border-none focus:ring-0 text-white p-4 resize-none overflow-y-auto text-sm placeholder:text-gray-500"
                rows={1}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isGenerating}
                className="p-3 mr-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-xl transition-colors shadow-lg shadow-indigo-500/20"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <div className="text-center mt-3 text-[11px] text-gray-500">
              AI models can make mistakes. Verify important information.
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
