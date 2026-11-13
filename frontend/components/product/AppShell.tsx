"use client";

import type { ReactNode } from "react";

export type ProductArea = "create" | "sets" | "live" | "brain" | "lab";

type NavItem = {
  id: ProductArea;
  label: string;
  href?: string;
  onClick?: () => void;
};

export function AppShell({
  active,
  accountLabel,
  onAccount,
  onOpenBrain,
  children,
}: {
  active: ProductArea;
  accountLabel: string;
  onAccount: () => void;
  onOpenBrain?: () => void;
  children: ReactNode;
}) {
  const nav: NavItem[] = [
    { id: "create", label: "Create", href: "/" },
    { id: "sets", label: "Sets", href: "/sets" },
    { id: "live", label: "Live", href: "/live" },
    { id: "brain", label: "Brain", onClick: onOpenBrain },
    { id: "lab", label: "Lab", href: "/lab" },
  ];

  return (
    <div className="product-shell">
      <aside className="product-shell__rail" aria-label="SAUND navigation">
        <a className="product-shell__brand" href="/" aria-label="SAUND.AI home">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </a>

        <nav className="product-shell__nav" aria-label="Primary">
          {nav.map((item) => {
            const className = item.id === active ? "is-active" : "";
            if (item.onClick) {
              return (
                <button key={item.id} type="button" className={className} onClick={item.onClick}>
                  {item.label}
                </button>
              );
            }
            return (
              <a key={item.id} className={className} href={item.href ?? "#"}>
                {item.label}
              </a>
            );
          })}
        </nav>

        <div className="product-shell__footer">
          <div className="product-shell__health"><i /> Engine ready</div>
          <button type="button" className="product-shell__account" onClick={onAccount}>
            {accountLabel}
          </button>
        </div>
      </aside>

      <div className="product-shell__content">{children}</div>
    </div>
  );
}
