"use client";

import { useState } from "react";
import { ProWorkspace } from "@/components/ProWorkspace";
import { SimpleExperience } from "@/components/SimpleExperience";
import { SessionProvider } from "@/lib/session";

function ExperienceRouter() {
  const [experience, setExperience] = useState<"create" | "studio">("create");

  return experience === "studio" ? (
    <ProWorkspace onExit={() => setExperience("create")} />
  ) : (
    <SimpleExperience onOpenStudio={() => setExperience("studio")} />
  );
}

export default function Home() {
  return (
    <SessionProvider>
      <ExperienceRouter />
    </SessionProvider>
  );
}
