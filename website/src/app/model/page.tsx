"use client";

import { Box, ArrowLeft, Cpu, Zap, Download, Server } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";

const models = [
    { name: "DeepSeek-R1-Distill-Qwen-1.5B", size: "1.1 GB", tier: 1, family: "DeepSeek", desc: "Ultra-fast reasoning model suitable for any hardware." },
    { name: "DeepSeek-R1-Distill-Llama-8B", size: "4.9 GB", tier: 2, family: "DeepSeek", desc: "High quality reasoning logic from DeepSeek." },
    { name: "Llama-3.2-3B-Instruct", size: "2.0 GB", tier: 1, family: "Meta", desc: "Meta's lightning fast 3B model for edge devices." },
    { name: "Llama-3.1-8B-Instruct", size: "4.7 GB", tier: 2, family: "Meta", desc: "The standard for open source 8B class models." },
    { name: "Mistral-7B-Instruct-v0.3", size: "4.1 GB", tier: 2, family: "Mistral", desc: "Highly capable and uncensored instruction tuning." },
    { name: "Qwen2.5-7B-Instruct", size: "4.5 GB", tier: 2, family: "Alibaba", desc: "Excellent coding and multilingual capabilities." },
    { name: "Phi-3-Mini-4k-Instruct", size: "2.4 GB", tier: 1, family: "Microsoft", desc: "Microsoft's small language model heavyweight." },
    { name: "Gemma-2-9B-It", size: "5.4 GB", tier: 3, family: "Google", desc: "Google's 9B powerhouse." },
];

export default function ModelCatalog() {
    return (
        <div className="min-h-screen bg-[#0b0e14] text-slate-200 overflow-x-hidden relative selection:bg-indigo-500/30 font-sans">
            {/* Background Gradients */}
            <div className="absolute top-0 inset-x-0 h-[600px] w-full bg-gradient-to-b from-purple-900/20 via-[#0b0e14] to-transparent pointer-events-none opacity-60 blur-3xl"></div>

            {/* Navbar Minimal */}
            <nav className="fixed top-0 inset-x-0 z-50 h-16 border-b border-white/5 bg-[#0b0e14]/70 backdrop-blur-md px-6 flex items-center">
                <Link href="/" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                    <ArrowLeft className="w-5 h-5" />
                    <span className="font-medium text-sm">Back to Home</span>
                </Link>
            </nav>

            <main className="pt-32 pb-20 px-6 max-w-6xl mx-auto relative z-10">
                <div className="text-center mb-16 space-y-4">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/20 mb-6"
                    >
                        <Box className="w-8 h-8 text-purple-400" />
                    </motion.div>
                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-4xl md:text-5xl font-black text-white"
                    >
                        Model Catalog
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="text-lg text-slate-400 max-w-2xl mx-auto"
                    >
                        Personal LLM supports over 27 GGUF and GGML format models out of the box.
                        Here is a preview of the supported local models you can download directly via the desktop app.
                    </motion.p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {models.map((mod, idx) => (
                        <motion.div
                            key={mod.name}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 * (idx % 3) }}
                            className="p-6 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all border-l-4 border-l-indigo-500 glass-panel flex flex-col h-full group"
                        >
                            <div className="flex justify-between items-start mb-4">
                                <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">{mod.family}</span>
                                <span className="text-xs font-medium bg-white/10 px-2 py-1 rounded-md text-slate-300">Tier {mod.tier}</span>
                            </div>
                            <h3 className="text-xl font-bold text-white mb-2">{mod.name}</h3>
                            <p className="text-sm text-slate-400 mb-6 flex-grow">{mod.desc}</p>

                            <div className="flex items-center justify-between pt-4 border-t border-white/5">
                                <div className="flex items-center gap-2 text-slate-500 text-sm">
                                    <Server className="w-4 h-4" /> {mod.size}
                                </div>
                                <div className="flex flex-col items-end">
                                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold flex items-center gap-1">
                                        <Zap className="w-3 h-3 text-amber-400" /> GPU Ready
                                    </span>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                <div className="mt-20 p-8 rounded-3xl bg-indigo-900/20 border border-indigo-500/20 text-center max-w-3xl mx-auto flex flex-col items-center">
                    <Cpu className="w-12 h-12 text-indigo-400 mb-4" />
                    <h2 className="text-2xl font-bold text-white mb-2">Hardware Tiers Explained</h2>
                    <p className="text-slate-400 mb-6">
                        <strong className="text-white">Tier 1:</strong> Runs on almost any PC (8GB+ RAM).<br />
                        <strong className="text-white">Tier 2:</strong> Requires a dedicated GPU or Apple Silicon (16GB+ RAM).<br />
                        <strong className="text-white">Tier 3:</strong> For high-end workstations (24GB+ VRAM/RAM).
                    </p>
                    <Link href="/">
                        <button className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold transition-all shadow-lg flex items-center gap-2">
                            <Download className="w-4 h-4" /> Download App to Access Models
                        </button>
                    </Link>
                </div>
            </main>
        </div>
    );
}
