"use client";

import { Lock, ArrowRight } from "lucide-react";
import { useLanguage } from "@/components/language_provider";
import { actionCardTranslations } from "@/lib/translations";

export default function ActionCard({ action, onClick, locked = false }) {
  const Icon = action.icon;
  const { language } = useLanguage();
  const t = actionCardTranslations[language] || actionCardTranslations.en;

  return (
    <button
      type="button"
      onClick={locked ? undefined : onClick}
      disabled={locked}
      className={`group relative overflow-hidden rounded-3xl border text-left transition-all duration-300 ${
        locked
          ? "cursor-not-allowed border-white/10 bg-white/5 opacity-80"
          : "cursor-pointer border-white/15 bg-white/10 hover:-translate-y-1 hover:border-white/25 hover:bg-white/15 hover:shadow-2xl"
      }`}
    >
      <div
        className={`absolute inset-0 ${
          locked
            ? "bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.06),transparent_35%)]"
            : "bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.18),transparent_35%)]"
        }`}
      />

      <div className="relative flex h-full flex-col p-6 md:p-7">
        <div className="mb-5 flex items-start justify-between">
          <div
            className={`flex h-14 w-14 items-center justify-center rounded-2xl border ${
              locked
                ? "border-white/10 bg-white/5 text-white/50"
                : "border-white/20 bg-white/10 text-white"
            }`}
          >
            <Icon className="h-6 w-6" />
          </div>

          {locked ? (
            <div className="flex items-center gap-2 rounded-full border border-amber-400/20 bg-amber-400/10 px-3 py-1 text-xs font-medium text-amber-200">
              <Lock className="h-3.5 w-3.5" />
              {t.accountRequired}
            </div>
          ) : (
            <div className="flex items-center gap-1 text-sm font-medium text-white/70 transition-transform duration-300 group-hover:translate-x-1 group-hover:text-white">
              {t.open}
              <ArrowRight className="h-4 w-4" />
            </div>
          )}
        </div>

        <div className="space-y-2">
          <h3
            className={`text-xl font-semibold tracking-tight ${
              locked ? "text-white/75" : "text-white"
            }`}
          >
            {action.name}
          </h3>
          <p
            className={`text-sm leading-6 ${
              locked ? "text-white/45" : "text-white/70"
            }`}
          >
            {action.description}
          </p>
        </div>
      </div>
    </button>
  );
}
