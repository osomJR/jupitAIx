"use client";

import { LanguageProvider } from "@/components/language_provider";

export default function Providers({ children, initialLanguage }) {
  return (
    <LanguageProvider initialLanguage={initialLanguage}>
      {children}
    </LanguageProvider>
  );
}
