"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Dispatch, ReactNode, SetStateAction } from "react";
import {
  getAuthSession,
  loginAccount,
  logoutAccount,
  registerAccount,
  updateAccountPreferences,
  updateAccountProfile,
} from "./api";
import type { AccountPreferences, AccountProfile, AccountRegistrationInput, AccountUser, AuthSession, JourneyMode } from "./types";

type StudioJourney = "balanced" | "smooth" | "build";

type SessionContextValue = {
  selectedPaths: string[];
  setSelectedPaths: Dispatch<SetStateAction<string[]>>;
  listener: string;
  createJourney: JourneyMode;
  setCreateJourney: Dispatch<SetStateAction<JourneyMode>>;
  studioJourney: StudioJourney;
  setStudioJourney: Dispatch<SetStateAction<StudioJourney>>;
  account: AccountUser | null;
  accountProfile: AccountProfile | null;
  accountPreferences: AccountPreferences | null;
  authReady: boolean;
  refreshAccount: () => Promise<AuthSession>;
  signIn: (email: string, password: string) => Promise<AuthSession>;
  createAccount: (input: AccountRegistrationInput) => Promise<AuthSession>;
  signOut: () => Promise<void>;
  setAccountPreferences: (input: Partial<AccountPreferences>) => Promise<AuthSession>;
  setAccountProfile: (input: Partial<AccountProfile>) => Promise<AuthSession>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [listener, setListener] = useState("web-local");
  const [createJourney, setCreateJourney] = useState<JourneyMode>("balanced");
  const [studioJourney, setStudioJourney] = useState<StudioJourney>("balanced");
  const [account, setAccount] = useState<AccountUser | null>(null);
  const [accountProfile, setAccountProfileState] = useState<AccountProfile | null>(null);
  const [accountPreferences, setAccountPreferencesState] = useState<AccountPreferences | null>(null);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    try {
      const key = "saund-listener-id";
      const legacyKey = "djai-listener-id";
      const existing = window.localStorage.getItem(key) ?? window.localStorage.getItem(legacyKey);
      if (existing) {
        window.localStorage.setItem(key, existing);
        setListener(existing);
        return;
      }
      const created = `web-${window.crypto.randomUUID()}`;
      window.localStorage.setItem(key, created);
      setListener(created);
    } catch {
      // Storage can be unavailable in locked-down/private browser contexts.
    }
  }, []);

  function applyAuthSession(session: AuthSession) {
    setAccount(session.authenticated ? session.user : null);
    setAccountProfileState(session.authenticated ? session.profile : null);
    setAccountPreferencesState(session.authenticated ? session.preferences : null);
    return session;
  }

  async function refreshAccount() {
    try {
      return applyAuthSession(await getAuthSession());
    } catch {
      const fallback: AuthSession = { authenticated: false, user: null, profile: null, preferences: null };
      applyAuthSession(fallback);
      return fallback;
    } finally {
      setAuthReady(true);
    }
  }

  useEffect(() => {
    void refreshAccount();
  }, []);

  async function signIn(email: string, password: string) {
    const session = applyAuthSession(await loginAccount({ email, password }));
    setAuthReady(true);
    return session;
  }

  async function createAccount(input: AccountRegistrationInput) {
    const session = applyAuthSession(await registerAccount(input));
    setAuthReady(true);
    return session;
  }

  async function signOut() {
    try {
      await logoutAccount();
    } finally {
      setAccount(null);
      setAccountProfileState(null);
      setAccountPreferencesState(null);
      setAuthReady(true);
    }
  }

  async function setAccountPreferences(input: Partial<AccountPreferences>) {
    const session = applyAuthSession(await updateAccountPreferences(input));
    return session;
  }

  async function setAccountProfile(input: Partial<AccountProfile>) {
    const session = applyAuthSession(await updateAccountProfile(input));
    return session;
  }

  const value = useMemo(
    () => ({
      selectedPaths,
      setSelectedPaths,
      listener,
      createJourney,
      setCreateJourney,
      studioJourney,
      setStudioJourney,
      account,
      accountProfile,
      accountPreferences,
      authReady,
      refreshAccount,
      signIn,
      createAccount,
      signOut,
      setAccountPreferences,
      setAccountProfile,
    }),
    [selectedPaths, listener, createJourney, studioJourney, account, accountProfile, accountPreferences, authReady],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSaundSession() {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSaundSession must be used inside SessionProvider");
  return value;
}
