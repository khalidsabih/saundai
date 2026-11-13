import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import "./legacy-simple.css";
import "./create-cinematic.css";
import "./legacy-instrument.css";
// Preserve the exact former globals.css cascade order after feature ownership split.
import "./training-lab.css";
import "./account-foundation.css";
import "./legacy-experiences.css";
import "./design-tokens.css";
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
