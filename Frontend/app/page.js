"use client";

import { useMemo } from "react";
import { useLanguage } from "@/components/language_provider";
import ActionCard from "@/components/ActionCard";
import { useRouter } from "next/navigation";
import {
  FileText,
  Sparkles,
  Languages,
  BookOpen,
  PenTool,
  Mic,
  HelpCircle,
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
};

export default function HomePage() {
  const router = useRouter();
  const { language, setLanguage } = useLanguage();

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
    <main className="min-h-screen bg-[#07111f] text-white">
      <div className="relative isolate overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_28%),radial-gradient(circle_at_top_right,rgba(168,85,247,0.18),transparent_30%),linear-gradient(to_bottom,#081120,#0a1426,#07111f)]" />

        <div className="relative mx-auto max-w-7xl px-6 py-12 md:px-8 md:py-16">
          <div className="mb-8 flex justify-end">
            <div className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 p-1 backdrop-blur">
              <span className="px-3 text-xs font-medium text-white/60">
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
                    : "text-white/70 hover:bg-white/10 hover:text-white"
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
              <p className="mt-5 max-w-2xl text-base leading-7 text-white/70 md:text-lg">
                {t.heroDescription}
              </p>
            </div>
          </section>

          <section>
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white md:text-2xl">
                  {t.availableNowTitle}
                </h2>
                <p className="mt-1 text-sm text-white/60">
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
            <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl md:p-8">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_left,rgba(34,211,238,0.18),transparent_25%),radial-gradient(circle_at_right,rgba(168,85,247,0.14),transparent_25%)]" />
              <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                <div className="max-w-2xl">
                  <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300/90">
                    {t.unlockMoreEyebrow}
                  </p>
                  <h3 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
                    {t.unlockMoreTitle}
                  </h3>
                  <p className="mt-3 text-sm leading-6 text-white/65 md:text-base">
                    {t.unlockMoreDescription}
                  </p>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => router.push("/signin")}
                    className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
                  >
                    {t.signIn}
                  </button>
                  <button
                    type="button"
                    onClick={() => router.push("/signup")}
                    className="rounded-2xl border border-white/15 bg-white/10 px-5 py-3 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15"
                  >
                    {t.signUp}
                  </button>
                </div>
              </div>
            </div>
          </section>

          <section>
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white/90 md:text-2xl">
                  {t.advancedFeaturesTitle}
                </h2>
                <p className="mt-1 text-sm text-white/50">
                  {t.advancedFeaturesDescription}
                </p>
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
              {lockedActions.map((action) => (
                <ActionCard
                  key={`${language}-${action.route}`}
                  action={action}
                  locked
                />
              ))}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
