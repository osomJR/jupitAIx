"use client";

export default function AuthControls({
  user,
  authChecked,
  signInLabel = "Sign In",
  signUpLabel = "Sign Up",
  loadingLabel = "Loading...",
  signedInAsLabel = "Signed in as",
  logoutLabel = "Logout",
}) {
  if (!authChecked) {
    return <div className="text-sm text-white/60">{loadingLabel}</div>;
  }

  if (!user) {
    return (
      <div className="flex flex-wrap gap-3">
        <a
          href="/auth/login?returnTo=/"
          className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-900 transition hover:scale-[1.02] hover:shadow-xl"
        >
          {signInLabel}
        </a>

        <a
          href="/auth/login?screen_hint=signup&prompt=login&returnTo=/"
          className="rounded-2xl border border-white/15 bg-white/10 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
        >
          {signUpLabel}
        </a>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="text-sm text-white/80">
        {signedInAsLabel}{" "}
        <span className="font-semibold">
          {user.name || user.email || user.nickname}
        </span>
      </div>

      <a
        href="/auth/logout"
        className="rounded-2xl border border-white/15 bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/15"
      >
        {logoutLabel}
      </a>
    </div>
  );
}