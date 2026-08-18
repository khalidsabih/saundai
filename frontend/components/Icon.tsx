import type { ReactNode, SVGProps } from "react";

export type IconName =
  | "library"
  | "route"
  | "pulse"
  | "render"
  | "live"
  | "upload"
  | "spark"
  | "close"
  | "drag"
  | "play"
  | "pause"
  | "download"
  | "warning"
  | "check"
  | "waveform"
  | "sliders"
  | "arrow"
  | "headphones"
  | "clock"
  | "shield"
  | "refresh"
  | "volume"
  | "mute";

export function Icon({
  name,
  ...props
}: { name: IconName } & SVGProps<SVGSVGElement>) {
  const common = {
    width: 20,
    height: 20,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  let content: ReactNode;

  switch (name) {
    case "library":
      content = (
        <>
          <path d="M4 5.5h16v13H4z" />
          <path d="M8 5.5v13M16 5.5v13" />
          <path d="M4 10h4m8 4h4" />
        </>
      );
      break;
    case "route":
      content = (
        <>
          <circle cx="5" cy="6" r="2" />
          <circle cx="19" cy="18" r="2" />
          <path d="M7 6h5a4 4 0 0 1 4 4v2a4 4 0 0 0 3 3.9" />
          <path d="m16 13 3 3-3 3" />
        </>
      );
      break;
    case "pulse":
      content = <path d="M3 12h3l2.2-5 3.2 10 2.2-6 2 4H21" />;
      break;
    case "render":
      content = (
        <>
          <path d="M4 4h16v16H4z" />
          <path d="m9 8 7 4-7 4z" />
        </>
      );
      break;
    case "live":
      content = (
        <>
          <path d="M8.5 8.5a5 5 0 0 0 0 7" />
          <path d="M5.5 5.5a9 9 0 0 0 0 13" />
          <circle cx="12" cy="12" r="2" />
          <path d="M15.5 8.5a5 5 0 0 1 0 7" />
          <path d="M18.5 5.5a9 9 0 0 1 0 13" />
        </>
      );
      break;
    case "upload":
      content = (
        <>
          <path d="M12 16V4" />
          <path d="m7.5 8.5 4.5-4.5 4.5 4.5" />
          <path d="M5 14v5h14v-5" />
        </>
      );
      break;
    case "spark":
      content = (
        <>
          <path d="m12 2 1.5 5.5L19 9l-5.5 1.5L12 16l-1.5-5.5L5 9l5.5-1.5z" />
          <path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" />
        </>
      );
      break;
    case "close":
      content = <path d="m7 7 10 10M17 7 7 17" />;
      break;
    case "drag":
      content = (
        <>
          {[6, 12, 18].flatMap((y) =>
            [9, 15].map((x) => (
              <circle
                key={`${x}-${y}`}
                cx={x}
                cy={y}
                r="1"
                fill="currentColor"
                stroke="none"
              />
            )),
          )}
        </>
      );
      break;
    case "play":
      content = <path d="m9 6 9 6-9 6z" />;
      break;
    case "pause":
      content = <path d="M9 6v12M15 6v12" />;
      break;
    case "download":
      content = (
        <>
          <path d="M12 4v11" />
          <path d="m8 11 4 4 4-4" />
          <path d="M5 19h14" />
        </>
      );
      break;
    case "warning":
      content = (
        <>
          <path d="m12 3 9 16H3z" />
          <path d="M12 9v4M12 16h.01" />
        </>
      );
      break;
    case "check":
      content = <path d="m5 12 4 4 10-10" />;
      break;
    case "waveform":
      content = <path d="M3 12h2l1-5 2 10 2-13 2 16 2-11 2 6 2-3h3" />;
      break;
    case "sliders":
      content = (
        <>
          <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
          <circle cx="16" cy="7" r="2" />
          <circle cx="8" cy="17" r="2" />
        </>
      );
      break;
    case "arrow":
      content = (
        <>
          <path d="M5 12h14" />
          <path d="m14 7 5 5-5 5" />
        </>
      );
      break;
    case "headphones":
      content = (
        <>
          <path d="M4 13v-1a8 8 0 0 1 16 0v1" />
          <path d="M4 13h4v7H6a2 2 0 0 1-2-2zM20 13h-4v7h2a2 2 0 0 0 2-2z" />
        </>
      );
      break;
    case "clock":
      content = (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3 2" />
        </>
      );
      break;
    case "shield":
      content = (
        <>
          <path d="M12 3 5 6v5c0 4.5 2.7 8 7 10 4.3-2 7-5.5 7-10V6z" />
          <path d="m9 12 2 2 4-4" />
        </>
      );
      break;
    case "refresh":
      content = (
        <>
          <path d="M20 7v5h-5" />
          <path d="M4 17v-5h5" />
          <path d="M6.1 9a7 7 0 0 1 11.8-2L20 9M4 15l2.1 2a7 7 0 0 0 11.8-2" />
        </>
      );
      break;
    case "volume":
      content = (
        <>
          <path d="M5 10v4h3l4 3V7l-4 3z" />
          <path d="M15 9.5a4 4 0 0 1 0 5" />
          <path d="M17.5 7a7 7 0 0 1 0 10" />
        </>
      );
      break;
    case "mute":
      content = (
        <>
          <path d="M5 10v4h3l4 3V7l-4 3z" />
          <path d="m16 10 4 4M20 10l-4 4" />
        </>
      );
      break;
  }

  return (
    <svg {...common} {...props}>
      {content}
    </svg>
  );
}
