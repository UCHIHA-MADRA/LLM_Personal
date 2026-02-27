"use client";

import { Terminal, Download, Shield, Cpu, Zap, Box, Smartphone, Globe, Code2, Server } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#0b0e14] text-slate-200 overflow-x-hidden relative selection:bg-indigo-500/30 font-sans">

      {/* Background Gradients (Desktop App Style) */}
      <div className="absolute top-0 inset-x-0 h-[600px] w-full bg-gradient-to-b from-indigo-900/20 via-[#0b0e14] to-transparent pointer-events-none opacity-60 blur-3xl"></div>

      {/* Navbar */}
      <nav className="fixed top-0 inset-x-0 z-50 h-16 border-b border-white/5 bg-[#0b0e14]/70 backdrop-blur-md px-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Cpu className="text-white w-4 h-4" />
          </div>
          <span className="font-bold tracking-tight text-white/90">Personal LLM</span>
        </div>
        <div className="flex items-center gap-6">
          <a href="#features" className="text-sm font-medium text-slate-400 hover:text-white transition-colors hidden sm:block">Features</a>
          <Link href="/model" className="text-sm font-medium text-slate-400 hover:text-white transition-colors hidden sm:block">Models</Link>
          <a href="https://github.com/UCHIHA-MADRA/LLM_Personal" target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-slate-400 hover:text-white transition-colors flex items-center gap-2 bg-white/5 py-1.5 px-3 rounded-full hover:bg-white/10">
            <Globe className="w-4 h-4" /> GitHub
          </a>
        </div>
      </nav>

      <main className="pt-32 pb-20 px-6 max-w-6xl mx-auto relative z-10 flex flex-col items-center">

        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto mt-12 mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-semibold tracking-wide uppercase mb-8"
          >
            <Zap className="w-3.5 h-3.5 fill-indigo-400" /> V2.0.0 Now supporting 27+ Local AI Models
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-6xl md:text-8xl font-black tracking-tighter text-white leading-[1.05] mb-6"
          >
            Your AI. <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              100% On Your PC.
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-lg md:text-xl text-slate-400 leading-relaxed max-w-2xl mx-auto font-medium"
          >
            A totally private, zero-latency desktop and mobile app that runs massive open-source models directly on your hardware. No cloud. No subscriptions. No data mining.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-10"
          >
            <div className="flex flex-col items-center gap-2 w-full sm:w-auto">
              <button className="w-full sm:w-auto px-8 py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold transition-all shadow-xl shadow-indigo-900/50 flex items-center justify-center gap-2 hover:-translate-y-1 group">
                <Download className="w-5 h-5 group-hover:animate-bounce" /> Download for Windows
              </button>
              <span className="text-[11px] text-slate-500 font-medium">Windows 10/11 • 64-bit • Direct .EXE</span>
            </div>
            <div className="flex flex-col items-center gap-2 w-full sm:w-auto mt-4 sm:mt-0">
              <a href="https://expo.dev/artifacts/eas/9DMWPB2LSUeQnUJi7DCr1f.apk" className="w-full sm:w-auto px-8 py-4 rounded-xl bg-[#1e2330] hover:bg-[#252b3b] border border-white/10 text-white font-bold transition-all flex items-center justify-center gap-2 hover:-translate-y-1">
                <Smartphone className="w-5 h-5" /> Get Android APK
              </a>
              <span className="text-[11px] text-slate-500 font-medium">Remote inference over Local WiFi</span>
            </div>
          </motion.div>
        </div>

        {/* Desktop UI Mockup */}
        <div className="w-full max-w-4xl rounded-2xl border border-white/10 bg-[#0B0E14] shadow-2xl overflow-hidden glass-panel relative mx-auto mb-32">
          {/* Faux Window Controls */}
          <div className="h-10 bg-white/5 border-b border-white/5 flex items-center px-4 gap-2">
            <div className="w-3 h-3 rounded-full bg-red-400/80"></div>
            <div className="w-3 h-3 rounded-full bg-amber-400/80"></div>
            <div className="w-3 h-3 rounded-full bg-green-400/80"></div>
            <div className="ml-4 text-xs text-slate-500 font-medium">Personal LLM</div>
          </div>
          <div className="p-8 flex items-start gap-4 h-[400px]">
            <div className="w-10 h-10 rounded-xl bg-[#1E2330] flex items-center justify-center border border-white/10 shrink-0">
              <Cpu className="text-indigo-400 w-5 h-5" />
            </div>
            <div className="bg-white/5 rounded-2xl rounded-tl-none p-4 max-w-[80%] border border-white/5">
              <p className="text-sm leading-relaxed text-slate-200">
                Hello! I am your Personal LLM running entirely on your local hardware. My weights are loaded into your RAM and your data never leaves this machine.
                <br /><br />
                I can help you write code, reason through complex problems, or summarize your private documents. How can I help you today?
              </p>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div id="features" className="w-full mb-32">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-3xl font-bold text-white">Why use Personal LLM?</h2>
            <p className="text-slate-400">Ditch the monthly subscription and take control of your AI.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon={<Shield className="text-indigo-400 h-6 w-6" />}
              title="Total Privacy"
              desc="No API calls, no analytics, no telemetry. Even works entirely offline in airplane mode."
            />
            <FeatureCard
              icon={<Box className="text-purple-400 h-6 w-6" />}
              title="Mega Model Catalog"
              desc="Hot-swap between 27+ top-tier open models like DeepSeek-R1, Llama 3, Mistral, and Qwen."
            />
            <FeatureCard
              icon={<Cpu className="text-amber-400 h-6 w-6" />}
              title="GPU Acceleration"
              desc="Automatically uses your NVIDIA, AMD, or Apple Silicon GPU for lighting-fast inference."
            />
            <FeatureCard
              icon={<Smartphone className="text-pink-400 h-6 w-6" />}
              title="Native Mobile App"
              desc="Access your PC's models directly from the companion Android app over your local network."
            />
            <FeatureCard
              icon={<Globe className="text-blue-400 h-6 w-6" />}
              title="Cloud Fallback"
              desc="Plug in your OpenAI or Groq API keys to mix local models with massive cloud models seamlessly."
            />
            <FeatureCard
              icon={<Terminal className="text-emerald-400 h-6 w-6" />}
              title="Open Source"
              desc="100% transparent Python and React codebase. Inspect, modify, or extend it yourself."
            />
          </div>
        </div>
      </main>

      <footer className="border-t border-white/5 py-12 text-center text-slate-500 text-sm">
        <p>Built for complete digital autonomy. ❤️</p>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="p-6 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors glass-panel group">
      <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
    </div>
  )
}
