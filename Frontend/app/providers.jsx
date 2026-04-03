"use client";

import { LanguageProvider } from "@/components/language_provider";
import { ThemeProvider } from "@/components/theme_provider";

export default function Providers({ children, initialLanguage }) {
  return (
    <LanguageProvider initialLanguage={initialLanguage}>
      <ThemeProvider>{children}</ThemeProvider>
    </LanguageProvider>
  );
}