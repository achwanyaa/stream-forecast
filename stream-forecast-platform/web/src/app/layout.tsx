import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'Stream Forecast Platform — Behavioral Diagnostics Dashboard',
  description: 'High-frequency multi-rate behavioral stream forecasting and diagnostics platform',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased selection:bg-sky-500 selection:text-white">
        {/* Top Navigation Bar */}
        <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur sticky top-0 z-50 px-6 py-3">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-7 w-7 rounded bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white text-sm shadow-md shadow-sky-500/20">
                S
              </div>
              <span className="font-semibold text-slate-100 tracking-tight text-base">
                Stream Forecast Platform
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-sky-400 border border-slate-700/60 font-mono">
                v2.0-research
              </span>
            </div>

            <nav className="flex items-center space-x-1 sm:space-x-2 text-sm">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-md text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors font-medium"
              >
                Leaderboard
              </Link>
              <Link
                href="/audit"
                className="px-3 py-1.5 rounded-md text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors font-medium"
              >
                Phase 1 Audit
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
          {children}
        </main>

        <footer className="border-t border-slate-900 bg-slate-950/80 py-4 px-6 text-center text-xs text-slate-500">
          Stream Forecast Platform · Zero-Leakage Behavioral Stream Forecasting Harness
        </footer>
      </body>
    </html>
  );
}
