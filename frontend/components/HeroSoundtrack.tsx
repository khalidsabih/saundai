"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Icon } from "./Icon";

const INTRO_TARGET = 0.78;
const BED_INTRO_TARGET = 0.44;
const BED_WORK_TARGET = 0.24;
const NATURAL_CROSSFADE_SECONDS = 3.4;

type SoundtrackPhase = "intro" | "workspace" | "off";

type Props = {
  phase: SoundtrackPhase;
  onSoundStart?: () => void;
};

export type HeroSoundtrackHandle = {
  start: () => void;
};

export const HeroSoundtrack = forwardRef<HeroSoundtrackHandle, Props>(function HeroSoundtrack(
  { phase, onSoundStart },
  ref,
) {
  const introRef = useRef<HTMLAudioElement | null>(null);
  const bedRef = useRef<HTMLAudioElement | null>(null);
  const phaseRef = useRef<SoundtrackPhase>(phase);
  const startedRef = useRef(false);
  const bedStartedRef = useRef(false);
  const masterRef = useRef(0);
  const workspaceMixRef = useRef(phase === "workspace" ? 1 : 0);
  const [started, setStarted] = useState(false);
  const [muted, setMuted] = useState(false);
  const [blocked, setBlocked] = useState(false);

  async function ensureBedPlaying() {
    const bed = bedRef.current;
    if (!bed || bedStartedRef.current) return;
    bed.currentTime = 0;
    bed.loop = true;
    try {
      await bed.play();
      bedStartedRef.current = true;
    } catch {
      // A user gesture or the intro's playback permission will retry this later.
    }
  }

  async function begin(fromAutoplay = false) {
    const intro = introRef.current;
    const bed = bedRef.current;
    if (!intro || !bed || startedRef.current || phaseRef.current === "off") return;

    intro.currentTime = 0;
    intro.volume = 0;
    bed.currentTime = 0;
    bed.volume = 0;
    masterRef.current = 0;
    workspaceMixRef.current = phaseRef.current === "workspace" ? 0.18 : 0;

    try {
      const introPlayback = intro.play();
      // Prime the looping bed inside the same permission window. It stays silent
      // until the opener hands over to it.
      void bed.play().then(() => {
        bedStartedRef.current = true;
      }).catch(() => undefined);
      await introPlayback;
      startedRef.current = true;
      setStarted(true);
      setMuted(false);
      setBlocked(false);
      onSoundStart?.();
    } catch {
      if (fromAutoplay) setBlocked(true);
    }
  }

  useImperativeHandle(ref, () => ({
    start: () => {
      void begin(false);
    },
  }));

  useEffect(() => {
    phaseRef.current = phase;
    if (phase === "workspace" && startedRef.current) void ensureBedPlaying();
  }, [phase]);

  useEffect(() => {
    const intro = introRef.current;
    const bed = bedRef.current;
    if (!intro || !bed) return;

    const tick = window.setInterval(() => {
      const currentPhase = phaseRef.current;
      const targetMaster = muted || currentPhase === "off" ? 0 : 1;
      masterRef.current += (targetMaster - masterRef.current) * 0.16;
      if (Math.abs(targetMaster - masterRef.current) < 0.004) masterRef.current = targetMaster;

      if (currentPhase === "workspace") {
        workspaceMixRef.current += (1 - workspaceMixRef.current) * 0.09;
      }

      const introDuration = Number.isFinite(intro.duration) ? intro.duration : 10.2;
      const remaining = Math.max(0, introDuration - intro.currentTime);
      const naturalCrossfade = startedRef.current && !intro.ended && remaining <= NATURAL_CROSSFADE_SECONDS;

      if ((currentPhase === "workspace" || naturalCrossfade) && !bedStartedRef.current) {
        void ensureBedPlaying();
      }

      let introMix = 0;
      let bedMix = 0;

      if (startedRef.current) {
        if (currentPhase === "workspace") {
          const progress = workspaceMixRef.current;
          introMix = INTRO_TARGET * (1 - progress);
          bedMix = BED_WORK_TARGET * progress;
        } else if (currentPhase === "intro") {
          introMix = INTRO_TARGET;
          if (intro.currentTime < 0.85) {
            introMix *= Math.max(0, intro.currentTime / 0.85);
          }

          if (naturalCrossfade) {
            const progress = 1 - (remaining / NATURAL_CROSSFADE_SECONDS);
            introMix *= 1 - progress;
            bedMix = BED_INTRO_TARGET * progress;
          } else if (intro.ended) {
            introMix = 0;
            bedMix = BED_INTRO_TARGET;
          }
        }
      }

      const master = masterRef.current;
      intro.volume = Math.max(0, Math.min(1, introMix * master));
      bed.volume = Math.max(0, Math.min(1, bedMix * master));

      if (currentPhase === "workspace" && workspaceMixRef.current > 0.985 && !intro.paused) {
        intro.pause();
      }

      if (currentPhase === "off" && master <= 0.01) {
        intro.pause();
        bed.pause();
      }
    }, 60);

    return () => window.clearInterval(tick);
  }, [muted]);

  useEffect(() => {
    if (phase !== "intro") return;
    const timer = window.setTimeout(() => void begin(true), 320);
    return () => window.clearTimeout(timer);
    // The opener gets one best-effort audible autoplay attempt. Browsers that block
    // it fall back to the explicit Sound on / Enter controls.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggleSound() {
    if (!started) {
      void begin(false);
      return;
    }
    setMuted((value) => !value);
  }

  function handleIntroEnded() {
    void ensureBedPlaying();
  }

  const soundOn = started && !muted && phase !== "off";
  const label = !started
    ? (blocked ? "Sound on" : "Play sound")
    : (muted ? "Sound on" : "Mute");

  return (
    <>
      <audio
        ref={introRef}
        src="/media/saund-hero-intro.mp3"
        preload="auto"
        onEnded={handleIntroEnded}
        aria-hidden="true"
      />
      <audio
        ref={bedRef}
        src="/media/saund-hero-bed.mp3"
        preload="metadata"
        loop
        aria-hidden="true"
      />
      {phase === "intro" && (
        <button
          type="button"
          className={`saund-sound-control ${soundOn ? "is-on" : ""}`}
          aria-label={label}
          aria-pressed={soundOn}
          onClick={toggleSound}
        >
          <Icon name={soundOn ? "volume" : "mute"} />
          <span>{label}</span>
        </button>
      )}
    </>
  );
});
