"use client";

import { useEffect, useMemo, useState } from "react";
import type { FormEvent, MouseEvent } from "react";
import { getAccountProjects, getHostingStatus } from "@/lib/api";
import { useSaundSession } from "@/lib/session";
import type {
  AccountExperienceLevel,
  AccountGenre,
  AccountMusicalPriority,
  AccountPrimaryUse,
  AccountProfile,
  AccountProject,
  AccountRegistrationInput,
} from "@/lib/types";
import { Icon } from "./Icon";

type AuthMode = "login" | "register";
type AuthReason = "account" | "build";

const GENRES: AccountGenre[] = ["House", "Techno", "Hip-Hop", "Drum & Bass", "Pop", "Afro", "Trance", "R&B", "Other"];
const USES: Array<{ value: AccountPrimaryUse; label: string }> = [
  { value: "personal", label: "Personal listening" },
  { value: "dj", label: "DJ / performer" },
  { value: "producer", label: "Music producer" },
  { value: "creator", label: "Content creator" },
  { value: "venue_event", label: "Venue / event" },
  { value: "other", label: "Other" },
];
const EXPERIENCE: Array<{ value: AccountExperienceLevel; label: string }> = [
  { value: "starting", label: "Just starting" },
  { value: "intermediate", label: "Intermediate" },
  { value: "experienced", label: "Experienced" },
  { value: "professional", label: "Professional" },
];
const PRIORITIES: Array<{ value: AccountMusicalPriority; label: string }> = [
  { value: "smooth_transitions", label: "Smooth transitions" },
  { value: "energy_progression", label: "Energy progression" },
  { value: "creative_transitions", label: "Creative transitions" },
  { value: "harmonic_mixing", label: "Harmonic mixing" },
  { value: "club_ready_sound", label: "Club-ready sound" },
  { value: "discovery", label: "Discovery" },
];

const EMPTY_PROFILE: AccountProfile = {
  first_name: "",
  last_name: "",
  artist_name: null,
  country: "",
  primary_use: null,
  experience_level: null,
  preferred_genres: [],
  musical_priorities: [],
};

function toggleValue<T extends string>(items: T[], value: T): T[] {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

export function AccountDialog({
  open,
  reason,
  onClose,
  onAuthenticated,
}: {
  open: boolean;
  reason: AuthReason;
  onClose: () => void;
  onAuthenticated?: () => void;
}) {
  const {
    account,
    accountProfile,
    accountPreferences,
    signIn,
    createAccount,
    signOut,
    setAccountPreferences,
    setAccountProfile,
  } = useSaundSession();
  const [mode, setMode] = useState<AuthMode>("login");
  const [registerStep, setRegisterStep] = useState(1);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [profile, setProfile] = useState<AccountProfile>(EMPTY_PROFILE);
  const [personalizationEnabled, setPersonalizationEnabled] = useState(false);
  const [improveSaundAi, setImproveSaundAi] = useState(false);
  const [marketingConsent, setMarketingConsent] = useState(false);
  const [inviteCode, setInviteCode] = useState("");
  const [signupMode, setSignupMode] = useState<"open" | "invite" | "closed">("open");
  const [busy, setBusy] = useState(false);
  const [preferenceBusy, setPreferenceBusy] = useState(false);
  const [profileBusy, setProfileBusy] = useState(false);
  const [editingProfile, setEditingProfile] = useState(false);
  const [projects, setProjects] = useState<AccountProject[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setPassword("");
    setConfirmPassword("");
    if (!account) {
      setMode("login");
      setRegisterStep(1);
    } else {
      setEditingProfile(false);
      setProfile(accountProfile ?? EMPTY_PROFILE);
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, account]);

  useEffect(() => {
    if (accountProfile && !editingProfile) setProfile(accountProfile);
  }, [accountProfile, editingProfile]);
  useEffect(() => {
    if (!open || account) return;
    let cancelled = false;
    void getHostingStatus().then((status) => {
      if (!cancelled) setSignupMode(status.signup_mode);
    }).catch(() => {
      if (!cancelled) setSignupMode("open");
    });
    return () => { cancelled = true; };
  }, [open, account]);


  useEffect(() => {
    if (!open || !account) {
      setProjects([]);
      return;
    }
    let cancelled = false;
    setProjectsLoading(true);
    void getAccountProjects(8)
      .then((payload) => {
        if (!cancelled) setProjects(payload.projects);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      })
      .finally(() => {
        if (!cancelled) setProjectsLoading(false);
      });
    return () => { cancelled = true; };
  }, [open, account]);

  const profileName = useMemo(() => {
    const named = [accountProfile?.first_name, accountProfile?.last_name].filter(Boolean).join(" ").trim();
    return named || account?.email || "Your account";
  }, [accountProfile, account]);

  if (!open) return null;

  function registrationStepValid(step: number) {
    if (step === 1) {
      if (!profile.first_name.trim() || !profile.last_name.trim() || !email.trim()) return "Add your name and email to continue.";
      if (password.length < 12) return "Use at least 12 characters for your password.";
      if (password !== confirmPassword) return "The passwords do not match.";
      if (signupMode === "invite" && !inviteCode.trim()) return "Enter your SAUND.AI beta invite code to continue.";
      if (signupMode === "closed") return "SAUND.AI registration is currently closed.";
    }
    if (step === 2) {
      if (!profile.country.trim()) return "Choose your country or region to continue.";
      if (!profile.primary_use) return "Choose how you plan to use SAUND.AI.";
    }
    return null;
  }

  function continueRegistration() {
    const validation = registrationStepValid(registerStep);
    if (validation) {
      setError(validation);
      return;
    }
    setError(null);
    setRegisterStep((current) => Math.min(4, current + 1));
  }

  async function submitLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(email, password);
      setPassword("");
      onAuthenticated?.();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  async function submitRegistration(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validation = registrationStepValid(1) ?? registrationStepValid(2);
    if (validation) {
      setError(validation);
      setRegisterStep(validation.includes("password") || validation.includes("name") || validation.includes("email") ? 1 : 2);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const input: AccountRegistrationInput = {
        ...profile,
        artist_name: profile.artist_name?.trim() || null,
        email: email.trim(),
        password,
        personalization_enabled: personalizationEnabled,
        improve_saund_ai: improveSaundAi,
        marketing_consent: marketingConsent,
        invite_code: inviteCode.trim() || null,
      };
      await createAccount(input);
      setPassword("");
      setConfirmPassword("");
      onAuthenticated?.();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  async function updatePreference(key: "personalization_enabled" | "improve_saund_ai" | "marketing_consent", value: boolean) {
    setPreferenceBusy(true);
    setError(null);
    try {
      await setAccountPreferences({ [key]: value });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setPreferenceBusy(false);
    }
  }

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!profile.first_name.trim() || !profile.last_name.trim() || !profile.country.trim() || !profile.primary_use) {
      setError("First name, last name, country, and primary use are required.");
      return;
    }
    setProfileBusy(true);
    setError(null);
    try {
      await setAccountProfile({
        ...profile,
        first_name: profile.first_name.trim(),
        last_name: profile.last_name.trim(),
        artist_name: profile.artist_name?.trim() || null,
        country: profile.country.trim(),
      });
      setEditingProfile(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setProfileBusy(false);
    }
  }

  async function handleSignOut() {
    setBusy(true);
    setError(null);
    try {
      await signOut();
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  function dismissFromBackdrop(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget && !busy && !profileBusy) onClose();
  }

  function renderProfileFields(includeName = true) {
    return (
      <>
        {includeName && (
          <div className="account-dialog__form-grid">
            <label>
              <span>First name *</span>
              <input value={profile.first_name} onChange={(event) => setProfile((current) => ({ ...current, first_name: event.target.value }))} autoComplete="given-name" required />
            </label>
            <label>
              <span>Last name *</span>
              <input value={profile.last_name} onChange={(event) => setProfile((current) => ({ ...current, last_name: event.target.value }))} autoComplete="family-name" required />
            </label>
          </div>
        )}
        <label>
          <span>Artist / DJ name</span>
          <input value={profile.artist_name ?? ""} onChange={(event) => setProfile((current) => ({ ...current, artist_name: event.target.value || null }))} autoComplete="nickname" placeholder="Optional" />
        </label>
        <label>
          <span>Country / region *</span>
          <input value={profile.country} onChange={(event) => setProfile((current) => ({ ...current, country: event.target.value }))} autoComplete="country-name" required />
        </label>
        <label>
          <span>Primary use *</span>
          <select value={profile.primary_use ?? ""} onChange={(event) => setProfile((current) => ({ ...current, primary_use: event.target.value as AccountPrimaryUse }))} required>
            <option value="" disabled>Choose one</option>
            {USES.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label>
          <span>Experience</span>
          <select value={profile.experience_level ?? ""} onChange={(event) => setProfile((current) => ({ ...current, experience_level: event.target.value ? event.target.value as AccountExperienceLevel : null }))}>
            <option value="">Prefer not to say</option>
            {EXPERIENCE.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
      </>
    );
  }

  function renderMusicChoices() {
    return (
      <>
        <div className="account-dialog__choice-group">
          <div className="account-dialog__choice-heading">
            <span>Preferred genres</span>
            <small>Optional · choose any that fit</small>
          </div>
          <div className="account-dialog__chips">
            {GENRES.map((genre) => (
              <button key={genre} type="button" className={profile.preferred_genres.includes(genre) ? "is-selected" : ""} onClick={() => setProfile((current) => ({ ...current, preferred_genres: toggleValue(current.preferred_genres, genre) }))}>
                {genre}
              </button>
            ))}
          </div>
        </div>
        <div className="account-dialog__choice-group">
          <div className="account-dialog__choice-heading">
            <span>What matters most?</span>
            <small>Optional</small>
          </div>
          <div className="account-dialog__chips account-dialog__chips--wide">
            {PRIORITIES.map((priority) => (
              <button key={priority.value} type="button" className={profile.musical_priorities.includes(priority.value) ? "is-selected" : ""} onClick={() => setProfile((current) => ({ ...current, musical_priorities: toggleValue(current.musical_priorities, priority.value) }))}>
                {priority.label}
              </button>
            ))}
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="account-dialog__backdrop" onMouseDown={dismissFromBackdrop}>
      <section className={`account-dialog ${mode === "register" && !account ? "account-dialog--onboarding" : ""}`} role="dialog" aria-modal="true" aria-labelledby="account-dialog-title">
        <button className="account-dialog__close" type="button" onClick={onClose} disabled={busy || profileBusy} aria-label="Close account dialog">
          <Icon name="close" />
        </button>

        {account ? (
          <>
            <header className="account-dialog__header">
              <span className="account-dialog__eyebrow">SAUND.AI account</span>
              <h2 id="account-dialog-title">{profileName}</h2>
              <p>{account.email}</p>
            </header>

            <section className="account-dialog__profile" aria-labelledby="account-profile-title">
              <div className="account-dialog__section-heading">
                <span id="account-profile-title">Profile</span>
                <button type="button" onClick={() => { setProfile(accountProfile ?? EMPTY_PROFILE); setEditingProfile((value) => !value); setError(null); }}>
                  {editingProfile ? "Cancel" : "Edit"}
                </button>
              </div>
              {editingProfile ? (
                <form className="account-dialog__form account-dialog__profile-form" onSubmit={saveProfile}>
                  {renderProfileFields()}
                  {renderMusicChoices()}
                  <button className="account-dialog__primary account-dialog__primary--full" type="submit" disabled={profileBusy}>
                    <span>{profileBusy ? "Saving…" : "Save profile"}</span>
                    <Icon name="arrow" />
                  </button>
                </form>
              ) : (
                <div className="account-dialog__profile-summary">
                  <span><small>Artist / DJ</small><strong>{accountProfile?.artist_name || "—"}</strong></span>
                  <span><small>Country</small><strong>{accountProfile?.country || "—"}</strong></span>
                  <span><small>Primary use</small><strong>{USES.find((item) => item.value === accountProfile?.primary_use)?.label || "—"}</strong></span>
                  <span><small>Experience</small><strong>{EXPERIENCE.find((item) => item.value === accountProfile?.experience_level)?.label || "—"}</strong></span>
                </div>
              )}
            </section>

            <section className="account-dialog__projects" aria-labelledby="account-projects-title">
              <div className="account-dialog__section-heading">
                <span id="account-projects-title">Recent sessions</span>
                <small>{projectsLoading ? "Loading…" : projects.length ? `${projects.length} recent` : "No saved sessions yet"}</small>
              </div>
              {!projectsLoading && projects.length > 0 && (
                <div className="account-dialog__project-list">
                  {projects.slice(0, 5).map((project) => (
                    <div className="account-dialog__project" key={project.project_id}>
                      <span>
                        <strong>{project.title}</strong>
                        <small>{project.track_count} tracks · {project.journey ?? "balanced"}</small>
                      </span>
                      <i data-status={project.status}>{project.status === "completed" ? "Ready" : project.status}</i>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <div className="account-dialog__settings">
              <label>
                <span><strong>Personalization</strong><small>Use your account preferences to shape future SAUND.AI experiences.</small></span>
                <input type="checkbox" checked={accountPreferences?.personalization_enabled ?? false} disabled={preferenceBusy} onChange={(event) => void updatePreference("personalization_enabled", event.target.checked)} />
              </label>
              <label>
                <span><strong>Improve SAUND.AI</strong><small>Allow eligible interactions and feedback to help improve models. This does not grant automatic use of uploaded audio.</small></span>
                <input type="checkbox" checked={accountPreferences?.improve_saund_ai ?? false} disabled={preferenceBusy} onChange={(event) => void updatePreference("improve_saund_ai", event.target.checked)} />
              </label>
              <label>
                <span><strong>Product updates</strong><small>Receive occasional SAUND.AI product news. Independent from model-improvement consent.</small></span>
                <input type="checkbox" checked={accountPreferences?.marketing_consent ?? false} disabled={preferenceBusy} onChange={(event) => void updatePreference("marketing_consent", event.target.checked)} />
              </label>
            </div>

            {error && <p className="account-dialog__error" role="alert">{error}</p>}

            <footer className="account-dialog__footer">
              <button type="button" className="account-dialog__secondary" onClick={() => void handleSignOut()} disabled={busy}>Log out</button>
              <button type="button" className="account-dialog__primary" onClick={onClose}>Done</button>
            </footer>
          </>
        ) : (
          <>
            <header className="account-dialog__header">
              <span className="account-dialog__eyebrow">{reason === "build" ? "One step before the session" : "SAUND.AI account"}</span>
              <h2 id="account-dialog-title">{mode === "register" ? "Your sound starts here." : reason === "build" ? "Log in to start." : "Welcome back."}</h2>
              <p>
                {reason === "build"
                  ? "Your tracks and direction stay exactly as they are. Log in or create an account, then SAUND.AI will prepare the session."
                  : mode === "register"
                    ? "A few details help SAUND.AI start with better context. Musical preferences are hints, not permanent labels."
                    : "Log in when you want to start sessions, save them, and later personalize your SAUND.AI experience."}
              </p>
            </header>

            <div className="account-dialog__tabs" role="tablist" aria-label="Account action">
              <button type="button" role="tab" aria-selected={mode === "login"} className={mode === "login" ? "is-active" : ""} onClick={() => { setMode("login"); setError(null); }}>Log in</button>
              <button type="button" role="tab" aria-selected={mode === "register"} className={mode === "register" ? "is-active" : ""} disabled={signupMode === "closed"} onClick={() => { setMode("register"); setRegisterStep(1); setError(null); }}>Create account</button>
            </div>

            {mode === "login" ? (
              <form className="account-dialog__form" onSubmit={submitLogin}>
                <label><span>Email</span><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" autoFocus required disabled={busy} /></label>
                <label><span>Password</span><input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" maxLength={256} required disabled={busy} /></label>
                {error && <p className="account-dialog__error" role="alert">{error}</p>}
                <button className="account-dialog__primary account-dialog__primary--full" type="submit" disabled={busy}>
                  <span>{busy ? "Please wait…" : reason === "build" ? "Log in & start" : "Log in"}</span><Icon name="arrow" />
                </button>
              </form>
            ) : (
              <form className="account-dialog__form account-dialog__register" onSubmit={submitRegistration}>
                <div className="account-dialog__steps" aria-label={`Create account step ${registerStep} of 4`}>
                  {[1, 2, 3, 4].map((step) => <span key={step} className={step <= registerStep ? "is-active" : ""}>{step}</span>)}
                </div>

                {registerStep === 1 && (
                  <div className="account-dialog__step">
                    <div className="account-dialog__step-heading"><span>01</span><div><strong>Your account</strong><small>The essentials.</small></div></div>
                    <div className="account-dialog__form-grid">
                      <label><span>First name *</span><input value={profile.first_name} onChange={(event) => setProfile((current) => ({ ...current, first_name: event.target.value }))} autoComplete="given-name" autoFocus required /></label>
                      <label><span>Last name *</span><input value={profile.last_name} onChange={(event) => setProfile((current) => ({ ...current, last_name: event.target.value }))} autoComplete="family-name" required /></label>
                    </div>
                    <label><span>Email *</span><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required /></label>
                    <label><span>Password *</span><input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" minLength={12} maxLength={256} required /><small>Use at least 12 characters.</small></label>
                    <label><span>Confirm password *</span><input type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} autoComplete="new-password" minLength={12} maxLength={256} required /></label>
                    {signupMode === "invite" && <label><span>Private beta invite *</span><input value={inviteCode} onChange={(event) => setInviteCode(event.target.value)} autoComplete="off" maxLength={128} required /><small>Use the invite code included with your SAUND.AI beta access.</small></label>}
                  </div>
                )}

                {registerStep === 2 && (
                  <div className="account-dialog__step">
                    <div className="account-dialog__step-heading"><span>02</span><div><strong>About you</strong><small>Enough context, no application form.</small></div></div>
                    {renderProfileFields(false)}
                  </div>
                )}

                {registerStep === 3 && (
                  <div className="account-dialog__step">
                    <div className="account-dialog__step-heading"><span>03</span><div><strong>Your sound</strong><small>Initial hints only. Your real taste can evolve later.</small></div></div>
                    {renderMusicChoices()}
                  </div>
                )}

                {registerStep === 4 && (
                  <div className="account-dialog__step">
                    <div className="account-dialog__step-heading"><span>04</span><div><strong>Preferences & privacy</strong><small>Each choice is independent and can be changed later.</small></div></div>
                    <div className="account-dialog__privacy-options">
                      <label><span><strong>Personalize my SAUND.AI experience</strong><small>Use account preferences to tailor your experience.</small></span><input type="checkbox" checked={personalizationEnabled} onChange={(event) => setPersonalizationEnabled(event.target.checked)} /></label>
                      <label><span><strong>Help improve SAUND.AI</strong><small>Use eligible interactions and feedback for model improvement. Off by default.</small></span><input type="checkbox" checked={improveSaundAi} onChange={(event) => setImproveSaundAi(event.target.checked)} /></label>
                      <label><span><strong>Product updates and news</strong><small>Occasional product communication. Off by default.</small></span><input type="checkbox" checked={marketingConsent} onChange={(event) => setMarketingConsent(event.target.checked)} /></label>
                    </div>
                    <p className="account-dialog__legal">By creating an account, you agree to SAUND.AI's Terms of Service and acknowledge the Privacy Policy.</p>
                  </div>
                )}

                {error && <p className="account-dialog__error" role="alert">{error}</p>}

                <div className="account-dialog__register-actions">
                  {registerStep > 1 ? <button className="account-dialog__secondary" type="button" onClick={() => { setRegisterStep((current) => current - 1); setError(null); }} disabled={busy}>Back</button> : <span />}
                  {registerStep < 4 ? (
                    <button className="account-dialog__primary" type="button" onClick={continueRegistration}>Continue</button>
                  ) : (
                    <button className="account-dialog__primary" type="submit" disabled={busy}>{busy ? "Creating…" : reason === "build" ? "Create account & build" : "Create account"}</button>
                  )}
                </div>
              </form>
            )}

            <footer className="account-dialog__guest">
              <span>You can keep choosing music without an account.</span>
              <button type="button" onClick={onClose} disabled={busy}>Not now</button>
            </footer>
          </>
        )}
      </section>
    </div>
  );
}
