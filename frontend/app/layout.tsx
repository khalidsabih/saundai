import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import "./autonomous.css";
import "./studio-intelligence.css";
import "./product-system.css";

export const metadata: Metadata = {
  title: "SAUND.AI — The autonomous DJ.",
  description:
    "An autonomous DJ that listens, thinks ahead, performs and adapts while the Executive protects every committed move.",
  applicationName: "SAUND.AI",
  icons: [{ rel: "icon", url: "/saund-mark.svg", type: "image/svg+xml" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
