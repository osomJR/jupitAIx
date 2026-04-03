"use client";

import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  Monitor,
  Moon,
  Sun,
  LogOut,
  Settings,
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

  const displayName = user?.name || user?.email || user?.nickname || "Account";
  const initial = displayName.trim().charAt(0).toUpperCase() || "A";

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => {
          setOpen((current) => !current);
          if (open) setShowSettings(false);
        }}
        className="flex items-center gap-3 rounded-2xl border app-surface px-3 py-2 text-sm app-text transition hover:bg-[var(--app-surface-strong)]"
      >
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[var(--app-button-bg)] text-sm font-semibold text-[var(--app-button-text)]">
          {initial}
        </div>
        <div className="hidden text-left sm:block">
          <div className="font-medium app-text">{displayName}</div>
        </div>
        <ChevronDown className="h-4 w-4 app-text-muted" />
      </button>

      {open ? (
        <div className="absolute right-0 z-50 mt-3 w-72 overflow-hidden rounded-3xl border app-surface-strong p-2 shadow-2xl backdrop-blur-xl">
          {!showSettings ? (
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => setShowSettings(true)}
                className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm app-text transition hover:bg-[var(--app-surface)]"
              >
                <Settings className="h-4 w-4 app-text-muted" />
                {settingsLabel}
              </button>

              <a
                href="/auth/logout"
                className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-sm app-text transition hover:bg-[var(--app-surface)]"
              >
                <LogOut className="h-4 w-4 app-text-muted" />
                {logoutLabel}
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
                    className={`flex w-full items-center justify-between rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "light"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      <Sun className="h-4 w-4" />
                      {lightLabel}
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme("dark")}
                    disabled={loading}
                    className={`flex w-full items-center justify-between rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "dark"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      <Moon className="h-4 w-4" />
                      {darkLabel}
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme("system")}
                    disabled={loading}
                    className={`flex w-full items-center justify-between rounded-2xl px-3 py-3 text-sm transition ${
                      theme === "system"
                        ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                        : "app-surface app-text hover:bg-[var(--app-surface-strong)]"
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      <Monitor className="h-4 w-4" />
                      {systemLabel}
                    </span>
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
