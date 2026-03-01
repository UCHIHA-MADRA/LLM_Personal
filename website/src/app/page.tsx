"use client";

import { Terminal, Download, Shield, Cpu, Zap, Smartphone, Globe, Apple, Monitor, Database, Lock, Settings } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [powerOn, setPowerOn] = useState(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="min-h-screen bg-[#e5e5e5] text-[#333333] overflow-x-hidden font-sans pb-20">

      {/* Skeuomorphic Navbar / Menu Bar */}
      <nav className="fixed top-0 inset-x-0 z-50 h-16 skeuo-nav px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Hardware Logo Button */}
          <div className="w-10 h-10 rounded-full bg-[#f0f0f0] border-2 border-white shadow-[0_2px_4px_rgba(0,0,0,0.1),inset_0_2px_4px_rgba(0,0,0,0.05)] flex items-center justify-center p-1 cursor-pointer hover:shadow-[0_1px_2px_rgba(0,0,0,0.1),inset_0_2px_6px_rgba(0,0,0,0.1)] active:scale-95 transition-all">
            <div className="w-full h-full rounded-full bg-gradient-to-b from-[#818cf8] to-[#4f46e5] flex items-center justify-center shadow-inner">
              <Cpu className="text-white w-4 h-4 shadow-sm" />
            </div>
          </div>
          <span className="font-bold tracking-tight text-xl text-[#444] skeuo-text drop-shadow-sm">Personal LLM</span>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex bg-[#e0e0e0] rounded-full p-1 border border-[#c0c0c0] shadow-[inset_0_1px_3px_rgba(0,0,0,0.1),0_1px_0_white]">
            <a href="#features" className="px-4 py-1.5 rounded-full text-sm font-bold text-[#555] hover:bg-[#d5d5d5] hover:shadow-inner transition-all hover:text-[#333] skeuo-text">Specs</a>
            <Link href="/model" className="px-4 py-1.5 rounded-full text-sm font-bold text-[#555] hover:bg-[#d5d5d5] hover:shadow-inner transition-all hover:text-[#333] skeuo-text">Models</Link>
          </div>

          <a href="https://github.com/UCHIHA-MADRA/LLM_Personal" target="_blank" rel="noopener noreferrer" className="skeuo-button h-10 px-4 rounded-lg flex items-center gap-2 font-bold text-sm">
            <Globe className="w-4 h-4 text-[#666]" />
            <span className="skeuo-text">Source</span>
          </a>
        </div>
      </nav>

      <main className="relative z-10 flex flex-col items-center pt-32 px-4 md:px-8 max-w-7xl mx-auto">

        {/* Hero Section Container (looks like a physical desk pad or device base) */}
        <div className="w-full mb-16 skeuo-panel rounded-3xl p-8 md:p-12 relative overflow-hidden">

          {/* Subtle grid lines like a cutting mat */}
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTE5IDE5SDBWMTBoMTl2OXoiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2Q1ZDVkNSIgc3Ryb2tlLW9wYWNpdHk9IjAuNSIvPjwvc3ZnPg==')] opacity-30 pointer-events-none"></div>

          <div className="flex flex-col lg:flex-row items-center gap-12 relative z-10">

            {/* Left Column: Copy & Actions */}
            <div className="flex-1 text-center lg:text-left space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-md skeuo-inset border border-[#c0c0c0] bg-[#e5e5e5]">
                <div className="w-2 h-2 rounded-full bg-green-500 shadow-[inset_0_1px_1px_rgba(0,0,0,0.5),0_0_5px_rgba(34,197,94,0.5)]"></div>
                <span className="text-xs font-bold text-[#666] uppercase tracking-wider skeuo-text">Version 2.1 Online</span>
              </div>

              <h1 className="text-5xl md:text-7xl font-black tracking-tighter text-[#222] leading-[1.05] skeuo-text space-y-2">
                <span className="block">Your Setup.</span>
                <span className="block text-[#4f46e5]">Your AI.</span>
              </h1>

              <p className="text-lg md:text-xl text-[#555] leading-relaxed max-w-xl mx-auto lg:mx-0 font-medium skeuo-text">
                Physical, uncompromising inference. Download open-source models straight to your hard drive. Zero telemetry.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-6 pt-4">
                {/* primary Download Button */}
                <div className="flex flex-col items-center lg:items-start gap-3 w-full sm:w-auto">
                  <a
                    href="https://github.com/UCHIHA-MADRA/LLM_Personal/releases"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-full sm:w-auto px-10 py-5 rounded-xl skeuo-button-primary font-bold text-lg flex items-center justify-center gap-3 group"
                  >
                    <div className="bg-white/20 p-1.5 rounded text-white shadow-inner">
                      <Download className="w-5 h-5 group-hover:translate-y-0.5 transition-transform" />
                    </div>
                    <span>Download Installer</span>
                  </a>
                  <div className="flex items-center gap-3 px-3 py-1.5 rounded-md skeuo-inset bg-[#e0e0e0] border border-[#ccc]">
                    <span className="text-xs font-bold text-[#666] flex items-center gap-1"><Monitor className="w-3.5 h-3.5" /> WIN</span>
                    <div className="w-px h-3 bg-[#ccc]"></div>
                    <span className="text-xs font-bold text-[#666] flex items-center gap-1"><Apple className="w-3.5 h-3.5" /> MAC</span>
                    <div className="w-px h-3 bg-[#ccc]"></div>
                    <span className="text-xs font-bold text-[#666] flex items-center gap-1"><Terminal className="w-3.5 h-3.5" /> LINUX</span>
                  </div>
                </div>

                {/* Secondary App Button */}
                <div className="flex flex-col items-center lg:items-start gap-3 w-full sm:w-auto mt-4 sm:mt-0">
                  <a
                    href="https://expo.dev/artifacts/eas/8DKCxfvW5a9864d5YUJBpc.apks"
                    className="w-full sm:w-auto px-8 py-5 rounded-xl skeuo-button font-bold text-lg flex items-center justify-center gap-3 group bg-[#f0f0f0]"
                  >
                    <div className="bg-[#e0e0e0] p-1.5 rounded text-[#444] border border-[#c0c0c0] shadow-inner drop-shadow-sm">
                      <Smartphone className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform" />
                    </div>
                    <span>Get Android APK</span>
                  </a>
                  <span className="text-xs font-bold text-[#777] skeuo-text ml-2 flex items-center gap-1.5">
                    <Zap className="w-3 h-3 text-[#f59e0b]" /> Local network pair
                  </span>
                </div>
              </div>
            </div>

            {/* Right Column: Physical Device Render */}
            <div className="w-full max-w-md lg:w-1/2 perspective-1000">
              {/* "Physical" Terminal Device */}
              <motion.div
                initial={{ rotateX: 10, rotateY: -10, y: 20 }}
                animate={{ rotateX: 0, rotateY: 0, y: 0 }}
                transition={{ duration: 1, type: "spring" }}
                className="w-full aspect-[4/3] rounded-2xl skeuo-metal p-5 flex flex-col shadow-2xl relative"
              >
                {/* Hardware details: Screws */}
                <div className="absolute top-3 left-3 w-2 h-2 rounded-full bg-[#bbb] border border-[#999] shadow-[inset_1px_1px_rgba(255,255,255,0.8),inset_-1px_-1px_2px_rgba(0,0,0,0.5)] flex items-center justify-center"><div className="w-[1px] h-full bg-[#888] rotate-45"></div></div>
                <div className="absolute top-3 right-3 w-2 h-2 rounded-full bg-[#bbb] border border-[#999] shadow-[inset_1px_1px_rgba(255,255,255,0.8),inset_-1px_-1px_2px_rgba(0,0,0,0.5)] flex items-center justify-center"><div className="w-[1px] h-full bg-[#888] -rotate-[30deg]"></div></div>
                <div className="absolute bottom-3 left-3 w-2 h-2 rounded-full bg-[#bbb] border border-[#999] shadow-[inset_1px_1px_rgba(255,255,255,0.8),inset_-1px_-1px_2px_rgba(0,0,0,0.5)] flex items-center justify-center"><div className="w-[1px] h-full bg-[#888] rotate-12"></div></div>
                <div className="absolute bottom-3 right-3 w-2 h-2 rounded-full bg-[#bbb] border border-[#999] shadow-[inset_1px_1px_rgba(255,255,255,0.8),inset_-1px_-1px_2px_rgba(0,0,0,0.5)] flex items-center justify-center"><div className="w-[1px] h-full bg-[#888] rotate-90"></div></div>

                {/* Device Bezel */}
                <div className="flex-1 leather-texture rounded-xl p-3 flex flex-col relative z-10 shadow-inner">
                  {/* Glass Screen */}
                  <div className={`flex-1 skeuo-screen p-4 flex flex-col font-mono text-sm transition-opacity duration-300 ${powerOn ? 'opacity-100' : 'opacity-0'}`}>
                    <div className="flex justify-between text-[#64748b] border-b border-[#334155] pb-2 mb-2 text-xs">
                      <span>root@personal-llm:~$</span>
                      <span>MEM: 14.2/32GB</span>
                    </div>
                    <div className="space-y-3 text-[#a7f3d0] flex-1 overflow-hidden">
                      <p><span className="text-[#93c5fd]">›</span> init model 'llama-3-8b-instruct'</p>
                      <p className="text-[#cbd5e1] text-xs">[OK] System memory mapped</p>
                      <p className="text-[#cbd5e1] text-xs">[OK] GPU layers: 33/33 (VRAM mapping)</p>
                      <br />
                      <p><span className="text-[#fca5a5]">User:</span> Summarize the RAG context architecture.</p>
                      <p className="mt-2 text-[#e2e8f0] relative pl-4 opacity-90 leading-relaxed font-sans text-xs">
                        {/* Status bar */}
                        <span className="absolute -left-1 top-0 bottom-0 w-[2px] bg-indigo-500 rounded-full"></span>
                        <span className="text-[#a7f3d0] font-mono mb-1 block">[Searching local vector database...]</span>
                        The Context Intelligence Engine uses ChromaDB to map document chunks. During inference, it retrieves the top-K relevant passages...<span className="inline-block w-1.5 h-3.5 bg-white ml-1 animate-pulse"></span>
                      </p>
                    </div>
                  </div>
                </div>

                {/* Bottom Hardware Controls */}
                <div className="h-14 mt-3 flex items-center justify-between px-4">
                  <div className="flex items-center gap-4">
                    {/* Power Switch */}
                    <button
                      onClick={() => setPowerOn(!powerOn)}
                      className={`w-12 h-6 rounded-full border-2 transition-colors relative shadow-inner flex items-center px-1 ${powerOn ? 'bg-green-500 border-green-600' : 'bg-[#a0a0a0] border-[#909090]'}`}
                    >
                      <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${powerOn ? 'translate-x-[20px]' : 'translate-x-0'}`}></div>
                    </button>
                    <span className="text-[10px] font-bold text-[#666] uppercase tracking-wider skeuo-text">PWR</span>
                  </div>

                  <div className="flex gap-2">
                    {/* Fake LED Indicators */}
                    <div className="flex flex-col items-center gap-1">
                      <div className={`w-3 h-3 rounded-full border border-black/20 shadow-inner ${powerOn ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)] animate-pulse' : 'bg-[#666]'} transition-colors`}></div>
                      <span className="text-[8px] font-bold text-[#888]">CPU</span>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                      <div className={`w-3 h-3 rounded-full border border-black/20 shadow-inner ${powerOn ? 'bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.6)] animate-[pulse_2s_infinite]' : 'bg-[#666]'} transition-colors`}></div>
                      <span className="text-[8px] font-bold text-[#888]">NET</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Features / Hardware Specs */}
        <div id="features" className="w-full">
          <div className="flex items-center gap-4 mb-8 pl-4">
            <h2 className="text-2xl font-black text-[#333] uppercase tracking-tight skeuo-text">Hardware Specs</h2>
            <div className="h-0.5 flex-1 bg-gradient-to-r from-[#ccc] to-transparent shadow-[0_1px_0_white]"></div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <HardwareModule
              icon={<Database className="w-6 h-6 text-[#555]" />}
              title="Local Vector Engine"
              desc="Drop PDFs and Code files into the app. They are parsed and stored locally in ChromaDB for instant Retrieval-Augmented Generation."
            />
            <HardwareModule
              icon={<Cpu className="w-6 h-6 text-[#555]" />}
              title="Iterative Reasoning"
              desc="Deep Think mode enables the model to create a self-correction loop, analyzing its own output for flaws before responding."
            />
            <HardwareModule
              icon={<Shield className="w-6 h-6 text-[#555]" />}
              title="Air-Gapped Security"
              desc="No API keys required. No internet required. 100% of your confidential data remains on your physical disk."
            />
            <HardwareModule
              icon={<Terminal className="w-6 h-6 text-[#555]" />}
              title="Python & React Core"
              desc="Built with enterprise-grade tech: FastAPI streaming backend, Next.js Desktop frontend, and React Native mobile."
            />
            <HardwareModule
              icon={<Settings className="w-6 h-6 text-[#555]" />}
              title="Context Adjustments"
              desc="Manually tweak context lengths, system prompts, self-refine depths, and GPU layer offloading directly in settings."
            />
            <HardwareModule
              icon={<Lock className="w-6 h-6 text-[#555]" />}
              title="GGUF Quantization"
              desc="Run models mathematically compressed to 4-bit or 8-bit precision, allowing 32B+ parameter models on consumer hardware."
            />
          </div>
        </div>
      </main>
    </div>
  );
}

function HardwareModule({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="p-6 rounded-2xl skeuo-panel group relative overflow-hidden flex flex-col h-full border border-white">

      {/* Decorative metal rivets */}
      <div className="absolute top-3 left-3 w-1.5 h-1.5 rounded-full bg-[#ccc] shadow-[inset_1px_1px_white,inset_-1px_-1px_rgba(0,0,0,0.2)]"></div>
      <div className="absolute top-3 right-3 w-1.5 h-1.5 rounded-full bg-[#ccc] shadow-[inset_1px_1px_white,inset_-1px_-1px_rgba(0,0,0,0.2)]"></div>

      <div className="w-14 h-14 rounded-xl skeuo-inset flex items-center justify-center mb-6 shadow-inner relative z-10 bg-[#e8e8e8]">
        {icon}
      </div>

      <div className="flex-1">
        <h3 className="text-xl font-bold text-[#333] mb-3 tracking-tight skeuo-text">{title}</h3>
        <p className="text-sm text-[#666] leading-relaxed font-medium">{desc}</p>
      </div>

      {/* Decorative barcode/serial number */}
      <div className="mt-6 pt-4 border-t border-[#d0d0d0] flex items-center justify-between shadow-[0_1px_0_white]">
        <div className="flex gap-[2px]">
          <div className="w-[1px] h-3 bg-[#999]"></div><div className="w-[3px] h-3 bg-[#999]"></div><div className="w-[1px] h-3 bg-[#999]"></div>
          <div className="w-[2px] h-3 bg-[#999]"></div><div className="w-[1px] h-3 bg-[#999]"></div><div className="w-[4px] h-3 bg-[#999]"></div>
          <div className="w-[1px] h-3 bg-[#999]"></div><div className="w-[2px] h-3 bg-[#999]"></div><div className="w-[1px] h-3 bg-[#999]"></div>
        </div>
        <span className="text-[9px] font-mono font-bold text-[#888]">{Math.random().toString(36).substring(2, 10).toUpperCase()}</span>
      </div>
    </div>
  )
}
