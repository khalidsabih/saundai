/**
 * Shared browser API transport.
 *
 * Keep wire-level behavior stable: same-origin default, no-store requests,
 * credential forwarding, JSON decoding, and HTTP error status/detail.
 * Feature endpoints and job lifecycle remain in api.ts.
 */
export const API_BASE = (process.env.NEXT_PUBLIC_SAUND_API_URL ?? "").replace(/\/$/, "");

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", credentials: "include", ...init });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload ?? `Request failed (${response.status})`;
    const error = new Error(typeof detail === "string" ? detail : detail.message ?? JSON.stringify(detail));
    Object.assign(error, { status: response.status, detail });
    throw error;
  }
  return payload as T;
}
