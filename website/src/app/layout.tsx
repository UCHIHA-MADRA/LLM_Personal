import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#4f46e5",
};

export const metadata: Metadata = {
  title: "Personal LLM — Your Private AI Assistant | 100% Offline",
  description:
    "Run powerful AI models entirely on your own hardware. No cloud, no API keys, no monthly fees. Chat, analyze documents, and generate code — all offline with zero data collection.",
  keywords: [
    "personal AI",
    "offline AI",
    "private LLM",
    "local AI assistant",
    "llama",
    "GGUF",
    "open source AI",
    "privacy AI",
    "self-hosted AI",
    "RAG",
    "document AI",
    "no cloud AI",
  ],
  authors: [{ name: "Prabh", url: "https://github.com/UCHIHA-MADRA" }],
  creator: "Prabh",
  publisher: "Personal LLM",
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://github.com/UCHIHA-MADRA/LLM_Personal",
    siteName: "Personal LLM",
    title: "Personal LLM — Your Private AI, On Your Hardware",
    description:
      "100% offline AI assistant. Run open-source LLMs locally with zero data collection. Available for Windows, macOS, Linux, and Android.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Personal LLM — Private AI Assistant",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Personal LLM — Your Private AI, On Your Hardware",
    description:
      "Run powerful AI models locally. No cloud, no tracking, no monthly fees. Open source.",
    images: ["/og-image.png"],
  },
  alternates: {
    canonical: "https://github.com/UCHIHA-MADRA/LLM_Personal",
  },
  category: "technology",
};

// JSON-LD structured data for search engines
const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "Personal LLM",
  description:
    "Fully offline, private AI assistant that runs open-source language models on your own hardware.",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Windows, macOS, Linux, Android",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  author: {
    "@type": "Person",
    name: "Prabh",
    url: "https://github.com/UCHIHA-MADRA",
  },
  license: "https://opensource.org/licenses/MIT",
  softwareVersion: "2.0.2",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
