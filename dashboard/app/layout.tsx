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
      <body>{children}</body>
    </html>
  );
}
