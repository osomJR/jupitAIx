"use client";
import { useEffect, useMemo, useState } from "react";
import { useLanguage } from "@/components/language_provider";
import ActionCard from "@/components/ActionCard";
import ProfileMenu from "@/components/profile_menu";
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
  LayoutDashboard,
  KeyRound,
  UsersRound,
  Menu,
  X,
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
  dashboard: LayoutDashboard,
  apiKeys: KeyRound,
  projectsTeam: UsersRound,
};

const sidebarActionKeys = [
  "summarize",
  "grammar",
  "translate",
  "explain",
  "questions",
];

const sidebarActionKeySet = new Set(sidebarActionKeys);

export default function HomePage() {
  const router = useRouter();
  const { language, setLanguage } = useLanguage();
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

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
      t.enabledActions
        .filter((action) => !sidebarActionKeySet.has(action.key))
        .map((action) => ({
          ...action,
          icon: actionIcons[action.key],
        })),
    [t],
  );

  const lockedActions = useMemo(
    () =>
      t.lockedActions
        .filter((action) => !sidebarActionKeySet.has(action.key))
        .map((action) => ({
          ...action,
          icon: actionIcons[action.key],
        })),
    [t],
  );

  const sidebarActions = useMemo(() => {
    const actionsByKey = new Map(
      [...t.enabledActions, ...t.lockedActions].map((action) => [
        action.key,
        {
          ...action,
          icon: actionIcons[action.key],
          requiresAuth: action.key === "questions",
        },
      ]),
    );

    return sidebarActionKeys
      .map((key) => actionsByKey.get(key))
      .filter(Boolean);
  }, [t]);

  const manageActions = useMemo(
    () =>
      (t.manageActions || []).map((action) => ({
        ...action,
        icon: actionIcons[action.key],
      })),
    [t],
  );


  const accountName =
    user?.name ||
    user?.fullName ||
    user?.displayName ||
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.email ||
    "Account";


  const avatarText =
    accountName
      ?.split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0])
      .join("")
      .toUpperCase() || "A";

  const requiresSignInLabel = t.requiresSignIn;

  function handleSidebarActionClick(action) {
    if (action.requiresAuth && !isSignedIn) {
      return;
    }

    router.push(action.route);
  }

  const sidebarActionList = (
    <div className={sidebarOpen ? "space-y-3" : "flex flex-col items-center gap-3"}>
      <nav
        aria-label={t.aiFeaturesTitle}
        className={sidebarOpen ? "space-y-0.5" : "flex flex-col items-center gap-1"}
      >
        <div
          className={
            sidebarOpen
              ? "mb-1 px-3 text-xs font-semibold normal-case tracking-[0.02em] app-text-soft"
              : "mb-1 text-center text-xs font-semibold normal-case tracking-[0.02em] app-text-soft"
          }
        >
          {sidebarOpen ? t.aiFeaturesTitle : t.aiFeaturesCompactTitle}
        </div>

        {sidebarActions.map((action) => {
          const Icon = action.icon;
          const requiresSignIn = action.requiresAuth && !isSignedIn;

          return (
            <button
              key={`${language}-sidebar-${action.route}`}
              type="button"
              onClick={() => handleSidebarActionClick(action)}
              aria-disabled={requiresSignIn}
              title={
                sidebarOpen
                  ? undefined
                  : requiresSignIn
                    ? `${action.name} - ${requiresSignInLabel}`
                    : action.name
              }
              className={`group flex rounded-xl text-sm transition ${
                sidebarOpen
                  ? "w-full items-center gap-2 px-3 py-1 text-left"
                  : "h-8 w-8 items-center justify-center"
              } ${
                requiresSignIn
                  ? "cursor-not-allowed opacity-60"
                  : "app-text hover:bg-[var(--app-surface-strong)]"
              }`}
            >
              <span className="relative flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border app-surface-strong">
                <Icon className="h-3.5 w-3.5" />
                {requiresSignIn ? (
                  <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-amber-400 ring-2 ring-[var(--app-bg)]" />
                ) : null}
              </span>

              {sidebarOpen ? (
                <span className="min-w-0 flex-1 leading-tight">
                  <span className="block truncate font-medium app-text">
                    {action.name}
                  </span>
                  {requiresSignIn ? (
                    <span className="block truncate text-[11px] app-text-soft">
                      {requiresSignInLabel}
                    </span>
                  ) : null}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>

      <nav
        aria-label={t.manageTitle}
        className={sidebarOpen ? "space-y-0.5" : "flex flex-col items-center gap-1"}
      >
        <div
          className={
            sidebarOpen
              ? "mb-1 px-3 text-xs font-semibold normal-case tracking-[0.02em] app-text-soft"
              : "mb-1 text-center text-xs font-semibold normal-case tracking-[0.02em] app-text-soft"
          }
        >
          {sidebarOpen ? t.manageTitle : t.manageCompactTitle}
        </div>

        {manageActions.map((action) => {
          const Icon = action.icon;
          const isDisabled = !action.route;

          return (
            <button
              key={`${language}-manage-${action.key}`}
              type="button"
              disabled={isDisabled}
              aria-disabled={isDisabled ? "true" : undefined}
              onClick={isDisabled ? undefined : () => router.push(action.route)}
              title={
                sidebarOpen
                  ? undefined
                  : isDisabled
                    ? `${action.name} - ${t.soon}`
                    : action.name
              }
              className={`group flex rounded-xl text-sm transition ${
                sidebarOpen
                  ? "w-full items-center gap-2 px-3 py-1 text-left"
                  : "h-8 w-8 items-center justify-center"
              } ${
                isDisabled
                  ? "cursor-not-allowed opacity-60"
                  : "app-text hover:bg-[var(--app-surface-strong)]"
              }`}
            >
              <span className="relative flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border app-surface-strong">
                <Icon className="h-3.5 w-3.5" />
              </span>

              {sidebarOpen ? (
                <span className="min-w-0 flex-1 leading-tight">
                  <span className="block truncate font-medium app-text">
                    {action.name}
                  </span>
                  {isDisabled ? (
                    <span className="block truncate text-[11px] app-text-soft">
                      {t.soon}
                    </span>
                  ) : null}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>
    </div>
  );


  const languageSwitcher = (
    <div className="rounded-2xl border app-surface-strong p-2.5 backdrop-blur">
      <p className="px-1 text-xs font-semibold normal-case tracking-[0.02em] app-text-soft">
        {t.languageLabel}
      </p>

      <div className="mt-2 grid gap-1.5">
        <button
          type="button"
          onClick={() => setLanguage("en")}
          className={`rounded-xl px-3 py-1.5 text-left text-sm font-medium transition ${
            language === "en"
              ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)] shadow-sm"
              : "app-text-muted hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
          }`}
        >
          {t.english}
        </button>

        <button
          type="button"
          onClick={() => setLanguage("fr")}
          className={`rounded-xl px-3 py-1.5 text-left text-sm font-medium transition ${
            language === "fr"
              ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)] shadow-sm"
              : "app-text-muted hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
          }`}
        >
          {t.french}
        </button>
      </div>
    </div>
  );


  return (
    <main className="app-shell min-h-screen bg-[var(--app-bg)] text-[var(--app-text)]">
      <aside
        className={`fixed left-0 top-0 z-50 flex h-dvh flex-col overflow-visible border-r app-surface backdrop-blur-xl transition-all duration-300 ${
          sidebarOpen ? "w-72" : "w-16"
        }`}
      >
        <div
          className={`flex min-h-12 items-center border-b border-[var(--app-border)] ${
            sidebarOpen ? "justify-between px-4" : "justify-center px-2"
          }`}
        >
          {sidebarOpen ? (
            <span className="text-sm font-semibold app-text">Menu</span>
          ) : null}

          <button
            type="button"
            onClick={() => setSidebarOpen((current) => !current)}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            className="inline-flex h-9 w-9 items-center justify-center rounded-xl app-text-muted transition hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
          >
            {sidebarOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </button>
        </div>

        <div
          className={`min-h-0 flex-1 overflow-y-auto ${
            sidebarOpen ? "px-3 py-2" : "px-2 py-2"
          }`}
        >
          {sidebarActionList}
        </div>

        <div
          className={`absolute bottom-3 left-3 right-3 overflow-visible ${
            sidebarOpen ? "" : "flex justify-center"
          }`}
        >
          {sidebarOpen ? (
            <div className="space-y-2">
              {languageSwitcher}

              <div className="border-t border-[var(--app-border)] pt-1.5">
                {isSignedIn ? (
                  <ProfileMenu
                    user={user}
                    settingsLabel={t.settings}
                    logoutLabel={t.logout}
                    appearanceLabel={t.appearance}
                    lightLabel={t.light}
                    darkLabel={t.dark}
                    systemLabel={t.systemDefault}
                    menuPlacement="top"
                    menuAlign="left"
                    fullWidth
                  />
                ) : (
                  <div className="home-sidebar-auth rounded-2xl border app-surface-strong p-2.5 backdrop-blur">
                    {!authChecked ? (
                      <div className="text-sm app-text-soft">{t.loading}</div>
                    ) : (
                      <div className="grid grid-cols-2 gap-2">
                        <a
                          href="/auth/login?returnTo=/"
                          className="rounded-xl bg-[var(--app-button-bg)] px-3 py-2 text-center text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.02] hover:shadow-xl"
                        >
                          {t.signIn}
                        </a>

                        <a
                          href="/auth/login?screen_hint=signup&prompt=login&returnTo=/"
                          className="rounded-xl border app-surface px-3 py-2 text-center text-sm font-semibold app-text transition hover:bg-[var(--app-surface-strong)]"
                        >
                          {t.signUp}
                        </a>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : isSignedIn ? (
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open account menu"
              className="flex h-10 w-10 items-center justify-center rounded-full bg-teal-500 text-[11px] font-semibold text-[var(--app-text)] shadow-lg shadow-black/20"
            >
              {avatarText}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              className="inline-flex h-10 w-10 items-center justify-center rounded-xl app-text-muted transition hover:bg-[var(--app-surface-strong)] hover:text-[var(--app-text)]"
            >
              <Menu className="h-5 w-5" />
            </button>
          )}
        </div>
      </aside>

      <div
        className={`relative isolate min-h-screen overflow-visible transition-[padding] duration-300 ${
          sidebarOpen ? "pl-72" : "pl-16"
        }`}
      >
        <div className="absolute inset-0 app-hero-overlay" />

        {!isSignedIn ? (
          <div className="absolute right-6 top-4 z-20 flex items-center gap-3 md:right-8">
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
        ) : null}

        <div className="relative mx-auto max-w-7xl px-6 py-12 md:px-8 md:py-16">
          <section className="mb-12 md:mb-14">
            <div className="inline-flex items-center gap-2 rounded-full border border-[var(--app-accent-border)] bg-[var(--app-accent-bg)] px-4 py-2 text-sm text-[var(--app-accent-text)] backdrop-blur">
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
            <div className="relative overflow-hidden rounded-3xl border app-surface-strong p-6 backdrop-blur-xl md:p-8">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_left,rgba(34,211,238,0.18),transparent_25%),radial-gradient(circle_at_right,rgba(168,85,247,0.14),transparent_25%)]" />
              <div className="relative max-w-2xl">
                <p className="mb-2 text-sm font-semibold normal-case tracking-[0.02em] text-cyan-300/90">
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
