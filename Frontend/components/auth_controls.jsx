"use client";

import { useEffect, useState } from "react";

export default function AuthControls() {
  const [user, setUser] = useState(undefined);
  const [loading, setLoading] = useState(true);

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
            setLoading(false);
          }
          return;
        }

        const data = await res.json();

        if (!cancelled) {
          setUser(data);
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setUser(null);
          setLoading(false);
        }
      }
    }

    loadProfile();

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <div className="text-sm text-white/60">Loading...</div>;
  }

  if (!user) {
    return (
      <div className="flex flex-wrap gap-3">
        <a
          href="/auth/login?returnTo=/"
          className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
        >
          Sign In
        </a>

        <a
          href="/auth/login?screen_hint=signup&prompt=login&returnTo=/"
          className="rounded-2xl border border-white/15 bg-white/10 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
        >
          Sign Up
        </a>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="text-sm text-white/80">
        Signed in as{" "}
        <span className="font-semibold">
          {user.name || user.email || user.nickname}
        </span>
      </div>

      <a
        href="/auth/logout"
        className="rounded-2xl border border-white/15 bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/15"
      >
        Logout
      </a>
    </div>
  );
}
