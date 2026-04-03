"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

const ThemeContext = createContext({
  theme: "system",
  resolvedTheme: "dark",
  setTheme: async () => {},
  loading: true,
});

function normalizeTheme(value) {
  return value === "light" || value === "dark" || value === "system"
    ? value
    : "system";
}

function getSystemTheme() {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyThemeToDocument(theme) {
  if (typeof document === "undefined") return "dark";

  const resolved = theme === "system" ? getSystemTheme() : theme;
  const root = document.documentElement;

  root.dataset.theme = resolved;
  root.classList.remove("light", "dark");
  root.classList.add(resolved);

  return resolved;
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState("system");
  const [resolvedTheme, setResolvedTheme] = useState("dark");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadTheme() {
      try {
        const res = await fetch("/api/account/me", {
          method: "GET",
          credentials: "include",
          cache: "no-store",
        });

        if (!res.ok) {
          throw new Error("Could not load account settings");
        }

        const data = await res.json();
        const nextTheme = normalizeTheme(data?.settings?.appearance);

        if (!cancelled) {
          setThemeState(nextTheme);
          const nextResolved = applyThemeToDocument(nextTheme);
          setResolvedTheme(nextResolved || "dark");
        }
      } catch {
        if (!cancelled) {
          const fallbackTheme = "system";
          setThemeState(fallbackTheme);
          const nextResolved = applyThemeToDocument(fallbackTheme);
          setResolvedTheme(nextResolved || "dark");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadTheme();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const handleChange = () => {
      if (theme === "system") {
        const nextResolved = applyThemeToDocument("system");
        setResolvedTheme(nextResolved || "dark");
      }
    };

    media.addEventListener?.("change", handleChange);
    media.addListener?.(handleChange);

    return () => {
      media.removeEventListener?.("change", handleChange);
      media.removeListener?.(handleChange);
    };
  }, [theme]);

  const setTheme = useCallback(async (nextTheme) => {
    const normalized = normalizeTheme(nextTheme);

    setThemeState(normalized);
    const nextResolved = applyThemeToDocument(normalized);
    setResolvedTheme(nextResolved || "dark");

    try {
      const res = await fetch("/api/account/settings", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          appearance: normalized,
        }),
      });

      if (!res.ok) {
        throw new Error("Could not save appearance setting");
      }
    } catch (error) {
      console.error(error);
    }
  }, []);

  const value = useMemo(
    () => ({
      theme,
      resolvedTheme,
      setTheme,
      loading,
    }),
    [theme, resolvedTheme, setTheme, loading],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
