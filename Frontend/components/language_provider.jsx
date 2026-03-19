"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useRouter } from "next/navigation";

const LanguageContext = createContext({
  language: "en",
  setLanguage: () => {},
});

const COOKIE_KEY = "homepage-language";
const ONE_YEAR_IN_SECONDS = 60 * 60 * 24 * 365;
function normalizeLanguage(value) {
  return value === "fr" ? "fr" : "en";
}

export function LanguageProvider({ children, initialLanguage = "en" }) {
  const router = useRouter();
  const [language, setLanguageState] = useState(
    normalizeLanguage(initialLanguage),
  );

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const setLanguage = useCallback(
    (nextLanguage) => {
      const normalized = normalizeLanguage(nextLanguage);

      setLanguageState((current) => {
        if (current === normalized) return current;
        return normalized;
      });

      document.cookie = [
        `${COOKIE_KEY}=${normalized}`,
        "Path=/",
        `Max-Age=${ONE_YEAR_IN_SECONDS}`,
        "SameSite=Lax",
      ].join("; ");

      document.documentElement.lang = normalized;

      router.refresh();
    },
    [router],
  );

  const value = useMemo(
    () => ({
      language,
      setLanguage,
    }),
    [language, setLanguage],
  );

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
