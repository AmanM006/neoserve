import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NeoServe - Arm serving cost/SLO optimizer",
  description:
    "Latency-vs-cost Pareto frontier and $/token savings for LLM serving on AWS Graviton4 (Neoverse V2).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  );
}
