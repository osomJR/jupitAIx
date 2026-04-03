"use client";
import { useEffect, useMemo, useState } from "react";
import { useLanguage } from "@/components/language_provider";
import ActionCard from "@/components/ActionCard";
import { useRouter } from "next/navigation";
import AuthControls from "@/components/auth_controls";
import {
  FileText,
  Sparkles,
  Languages,
  BookOpen,
  PenTool,
  Mic,
  HelpCircle,
  EyeOff,
  EyeClosed,
  ShieldCheck,
  FileBraces,
} from "lucide-react";
import { homePageTranslations } from "@/lib/translations";
const actionIcons = {
  convert: FileText,
  summarize: Sparkles,
  grammar: PenTool,
  translate: Languages,
  explain: BookOpen,
  transcribe: Mic,
  questions: HelpCircle,
  redact: EyeOff,
  mask: EyeClosed,
  compliance: ShieldCheck,
  extraction: FileBraces,
};

export default function HomePage() {
  const router = useRouter();
  const { language, setLanguage } = useLanguage();
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      try {
        const res = await fetch("/auth/profile", {
          method: "GET",
          credentials: "include",
          cache: "no-store",
        });

        if (!res.ok) {
          if (!cancelled) {
            setUser(null);
            setAuthChecked(true);
          }
          return;
        }

        const data = await res.json();

        if (!cancelled) {
          setUser(data);
          setAuthChecked(true);
        }
      } catch {
        if (!cancelled) {
          setUser(null);
          setAuthChecked(true);
        }
      }
    }

    loadProfile();

    return () => {
      cancelled = true;
    };
  }, []);

  const isSignedIn = !!user;

  const t = useMemo(
    () => homePageTranslations[language] || homePageTranslations.en,
    [language],
  );

  const enabledActions = useMemo(
    () =>
      t.enabledActions.map((action) => ({
        ...action,
        icon: actionIcons[action.key],
      })),
    [t],
  );

  const lockedActions = useMemo(
    () =>
      t.lockedActions.map((action) => ({
        ...action,
        icon: actionIcons[action.key],
      })),
    [t],
  );

  return (
    <main className="app-shell">
      <div className="relative isolate overflow-visible">
        <div className="absolute inset-0 app-hero-overlay" />

        <div className="relative mx-auto max-w-7xl px-6 py-12 md:px-8 md:py-16">
          <div className="mb-8 flex justify-end">
            <div className="inline-flex items-center gap-2 rounded-2xl border app-surface p-1 backdrop-blur">
              <span className="px-3 text-xs font-medium app-text-soft">
                {t.languageLabel}
              </span>

              <button
                type="button"
                onClick={() => setLanguage("en")}
                className={`rounded-xl px-3 py-2 text-sm font-medium transition ${
                  language === "en"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-white/70 hover:bg-white/10 hover:text-white"
                }`}
              >
                {t.english}
              </button>

              <button
                type="button"
                onClick={() => setLanguage("fr")}
                className={`rounded-xl px-3 py-2 text-sm font-medium transition ${
                  language === "fr"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "app-text-muted hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
                }`}
              >
                {t.french}
              </button>
            </div>
          </div>

          <section className="mb-12 md:mb-14">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-200 backdrop-blur">
              <Sparkles className="h-4 w-4" />
              {t.badge}
            </div>

            <div className="mt-6 max-w-3xl">
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl md:text-6xl">
                {t.heroTitleStart}{" "}
                <span className="bg-gradient-to-r from-cyan-300 via-sky-400 to-violet-400 bg-clip-text text-transparent">
                  {t.heroTitleHighlight}
                </span>
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 app-text-muted md:text-lg">
                {t.heroDescription}
              </p>
            </div>
          </section>

          <section>
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold app-text md:text-2xl">
                  {t.availableNowTitle}
                </h2>
                <p className="mt-1 text-sm app-text-soft">
                  {t.availableNowDescription}
                </p>
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
              {enabledActions.map((action) => (
                <ActionCard
                  key={`${language}-${action.route}`}
                  action={action}
                  onClick={() => router.push(action.route)}
                />
              ))}
            </div>
          </section>

          <section className="my-12 md:my-14">
            <div className="relative overflow-visible rounded-3xl border app-surface-strong p-6 backdrop-blur-xl md:p-8">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_left,rgba(34,211,238,0.18),transparent_25%),radial-gradient(circle_at_right,rgba(168,85,247,0.14),transparent_25%)]" />
              <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                <div className="max-w-2xl">
                  <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300/90">
                    {t.unlockMoreEyebrow}
                  </p>

                  <h3 className="text-2xl font-semibold tracking-tight app-text md:text-3xl">
                    {isSignedIn ? t.unlockedSignedInTitle : t.unlockMoreTitle}
                  </h3>

                  <p className="mt-3 text-sm leading-6 app-text-muted md:text-base">
                    {isSignedIn
                      ? t.unlockedSignedInDescription
                      : t.unlockMoreDescription}
                  </p>
                </div>

                <div className="flex flex-wrap gap-3 overflow-visible">
                  <AuthControls
                    user={user}
                    authChecked={authChecked}
                    signInLabel={t.signIn}
                    signUpLabel={t.signUp}
                    loadingLabel={t.loading}
                    logoutLabel={t.logout}
                    settingsLabel={t.settings}
                    appearanceLabel={t.appearance}
                    lightLabel={t.light}
                    darkLabel={t.dark}
                    systemLabel={t.systemDefault}
                  />
                </div>
              </div>
            </div>
          </section>

          <section>
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold app-text md:text-2xl">
                  {t.advancedFeaturesTitle}
                </h2>
                <p className="mt-1 text-sm app-text-soft">
                  {authChecked
                    ? isSignedIn
                      ? t.advancedFeaturesSignedInDescription
                      : t.advancedFeaturesDescription
                    : t.checkingAccountStatus}
                </p>
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
              {lockedActions.map((action) => (
                <ActionCard
                  key={`${language}-${action.route}`}
                  action={action}
                  locked={!isSignedIn}
                  onClick={
                    isSignedIn ? () => router.push(action.route) : undefined
                  }
                />
              ))}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
