"use client";

import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  LogOut,
  Monitor,
  Moon,
  Settings,
  Sun,
} from "lucide-react";
import { useTheme } from "@/components/theme_provider";

export default function ProfileMenu({
  user,
  settingsLabel = "Settings",
  logoutLabel = "Logout",
  appearanceLabel = "Appearance",
  lightLabel = "Light",
  darkLabel = "Dark",
  systemLabel = "System Default",
  menuPlacement = "bottom",
  menuAlign = "right",
  fullWidth = false,
}) {
  const [open, setOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const containerRef = useRef(null);
  const { theme, setTheme, loading } = useTheme();

  useEffect(() => {
    function handleClickOutside(event) {
      if (!containerRef.current?.contains(event.target)) {
        setOpen(false);
        setShowSettings(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const displayName = user?.name || user?.nickname || user?.email || "Account";
  const displayEmail = user?.email || "";
  const initial = displayName.trim().charAt(0).toUpperCase() || "A";

  const menuPlacementClass =
    menuPlacement === "top" ? "bottom-full mb-3" : "top-full mt-3";

  const menuAlignClass = menuAlign === "left" ? "left-0" : "right-0";

  return (
    <div className={`relative ${fullWidth ? "w-full" : ""}`} ref={containerRef}>
      <button
        type="button"
        onClick={() => {
          setOpen((current) => !current);
          if (open) setShowSettings(false);
        }}
        className={`flex items-center gap-3 rounded-2xl border app-surface px-3 py-2 text-sm app-text transition hover:bg-[var(--app-surface-strong)] ${
          fullWidth ? "w-full justify-between" : ""
        }`}
      >
        <span className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[var(--app-button-bg)] text-sm font-semibold text-[var(--app-button-text)]">
            {initial}
          </span>

          <span className="min-w-0 text-left">
            <span className="block truncate font-medium app-text">
              {displayName}
            </span>
            {displayEmail ? (
              <span className="block truncate text-xs app-text-muted">
                {displayEmail}
              </span>
            ) : null}
          </span>
        </span>

        <ChevronDown className="h-4 w-4 shrink-0 app-text-muted" />
      </button>

      {open ? (
        <div
          className={`absolute ${menuAlignClass} ${menuPlacementClass} z-[80] w-72 overflow-hidden rounded-3xl border app-surface-strong p-2 shadow-2xl backdrop-blur-xl`}
        >
          {!showSettings ? (
            <div className="space-y-1">
              <div className="flex items-center gap-3 rounded-2xl px-3 py-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--app-button-bg)] text-sm font-semibold text-[var(--app-button-text)]">
                  {initial}
                </div>

                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold app-text">
                    {displayName}
                  </div>
                  {displayEmail ? (
                    <div className="truncate text-xs app-text-muted">
                      {displayEmail}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="my-2 h-px bg-white/10" />

              <button
                type="button"
                onClick={() => setShowSettings(true)}
                className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm app-text transition hover:bg-[var(--app-surface)]"
              >
                <Settings className="h-4 w-4 app-text-muted" />
                <span>{settingsLabel}</span>
              </button>

              <a
                href="/auth/logout"
                className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-sm app-text transition hover:bg-[var(--app-surface)]"
              >
                <LogOut className="h-4 w-4 app-text-muted" />
                <span>{logoutLabel}</span>
              </a>
            </div>
          ) : (
            <div className="space-y-2">
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="rounded-2xl px-3 py-2 text-sm app-text-muted transition hover:bg-[var(--app-surface)] hover:text-[var(--app-text)]"
              >
                ← {settingsLabel}
              </button>

              <div className="rounded-2xl border app-surface p-4">
                <div className="mb-3 text-sm font-semibold app-text">
                  {appearanceLabel}
                </div>

                <div className="space-y-2">
                  <button
                    type="button"
                    onClick={() => setTheme("light")}
                    disabled={loading}
                    className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "light"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <Sun className="h-4 w-4" />
                    {lightLabel}
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme("dark")}
                    disabled={loading}
                    className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "dark"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <Moon className="h-4 w-4" />
                    {darkLabel}
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme("system")}
                    disabled={loading}
                    className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "system"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <Monitor className="h-4 w-4" />
                    {systemLabel}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}