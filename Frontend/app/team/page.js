"use client";

import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, RefreshCw, Send, Trash2, UsersRound } from "lucide-react";
import { useAccount } from "@/components/account_provider";
import { useLanguage } from "@/components/language_provider";
import { teamPageTranslations } from "@/lib/translations";



function titleCase(value) {
  if (!value) return "—";
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function readJson(response) {
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      data?.detail?.message ||
        data?.detail?.error ||
        data?.error?.message ||
        "Request failed",
    );
  }

  return data;
}

export default function TeamPage() {
  const { language } = useLanguage();
  const { entitlement, reload: reloadAccount } = useAccount();
  const t = teamPageTranslations[language] || teamPageTranslations.en;

  const [loading, setLoading] = useState(true);
  const [organizations, setOrganizations] = useState([]);
  const [userInvitations, setUserInvitations] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [details, setDetails] = useState(null);
  const [subscription, setSubscription] = useState(null);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("member");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");

  const selectedOrganization = useMemo(
    () => organizations.find((org) => org.id === selectedId) || organizations[0] || null,
    [organizations, selectedId],
  );

  const members = useMemo(
    () => (details?.members || []).filter((member) => member.status === "active"),
    [details],
  );

  const pendingMemberInvitations = useMemo(
    () => (details?.members || []).filter((member) => member.status === "invited"),
    [details],
  );

  const canManage = ["owner", "admin"].includes(selectedOrganization?.member?.role);
  const seatsUsed = subscription?.active_members ?? members.length;
  const maxSeats = subscription?.max_accounts ?? null;
  const hasSeatLimit = typeof maxSeats === "number";
  const seatsAreFull = hasSeatLimit && seatsUsed >= maxSeats;
  const canInvite = canManage && !seatsAreFull;

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      credentials: "include",
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });

    return readJson(response);
  }

  async function loadOrganization(organizationId) {
    const [organizationDetails, subscriptionData] = await Promise.all([
      api(`/api/organizations/${organizationId}`),
      api(`/api/organizations/${organizationId}/subscription`),
    ]);

    setDetails(organizationDetails);
    setSubscription(subscriptionData.subscription);
  }

  async function load() {
    setLoading(true);
    setMessage("");

    try {
      const data = await api("/api/organizations/me");
      const orgs = data.organizations || [];
      const invitations = data.invitations || [];

      setOrganizations(orgs);
      setUserInvitations(invitations);

      const next =
        orgs.find((org) => org.id === entitlement?.organization_id) ||
        orgs[0] ||
        null;

      setSelectedId(next?.id || null);

      if (next?.id) {
        await loadOrganization(next.id);
      } else {
        setDetails(null);
        setSubscription(null);
      }
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function changeOrganization(event) {
    const organizationId = Number(event.target.value);
    setSelectedId(organizationId);
    setLoading(true);

    try {
      await loadOrganization(organizationId);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function acceptInvitation(organizationId) {
    setBusy(`accept:${organizationId}`);
    setMessage("");

    try {
      await api(`/api/organizations/${organizationId}/invitations/accept`, {
        method: "POST",
      });

      setMessage(t.acceptedInvitation);
      await reloadAccount();
      await load();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy("");
    }
  }

  async function inviteMember(event) {
    event.preventDefault();

    if (!selectedOrganization?.id || !email.trim()) {
      return;
    }

    if (seatsAreFull) {
      setMessage(t.upgradeRequiredDescription);
      return;
    }

    setBusy("invite");
    setMessage("");

    try {
      await api(`/api/organizations/${selectedOrganization.id}/members`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), role }),
      });

      setEmail("");
      setRole("member");
      await loadOrganization(selectedOrganization.id);
      await reloadAccount();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy("");
    }
  }

  async function updateRole(member, nextRole) {
    if (!selectedOrganization?.id) {
      return;
    }

    setBusy(`role:${member.user_id}`);
    setMessage("");

    try {
      await api(
        `/api/organizations/${selectedOrganization.id}/members/${encodeURIComponent(member.user_id)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: nextRole }),
        },
      );

      await loadOrganization(selectedOrganization.id);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy("");
    }
  }

  async function removeMember(member) {
    if (!selectedOrganization?.id) {
      return;
    }

    setBusy(`remove:${member.user_id}`);
    setMessage("");

    try {
      await api(
        `/api/organizations/${selectedOrganization.id}/members/${encodeURIComponent(member.user_id)}`,
        { method: "DELETE" },
      );

      await loadOrganization(selectedOrganization.id);
      await reloadAccount();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy("");
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entitlement?.organization_id]);

  if (loading && !selectedOrganization && !userInvitations.length) {
    return (
      <main className="min-h-screen app-page px-6 py-10">
        <div className="mx-auto max-w-6xl rounded-3xl border app-surface-strong p-8 app-text">
          {t.loading}
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen app-page px-6 py-10">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex flex-col gap-4 rounded-3xl border app-surface-strong p-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[var(--app-border)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] app-text-soft">
              <UsersRound className="h-3.5 w-3.5" />
              {t.organization}
            </div>
            <h1 className="text-3xl font-semibold tracking-tight app-text">
              {t.title}
            </h1>
            <p className="mt-2 max-w-2xl text-sm app-text-muted">
              {t.subtitle}
            </p>
          </div>

          <button
            type="button"
            onClick={load}
            className="inline-flex items-center justify-center gap-2 rounded-2xl border app-surface px-4 py-3 text-sm font-semibold app-text transition hover:bg-[var(--app-button-bg)] hover:text-[var(--app-button-text)]"
          >
            <RefreshCw className="h-4 w-4" />
            {t.refresh}
          </button>
        </header>

        {message ? (
          <div className="rounded-2xl border border-[var(--app-border)] app-surface-strong px-4 py-3 text-sm app-text">
            {message}
          </div>
        ) : null}

        {userInvitations.length ? (
          <section className="rounded-3xl border app-surface-strong p-6">
            <h2 className="text-xl font-semibold app-text">
              {t.invitationsForYou}
            </h2>
            <p className="mt-2 text-sm app-text-muted">
              {t.invitationsForYouDescription}
            </p>

            <div className="mt-5 grid gap-3 md:grid-cols-2">
              {userInvitations.map((invitation) => (
                <div
                  key={invitation.id}
                  className="rounded-2xl border app-surface p-4"
                >
                  <div className="text-sm font-semibold app-text">
                    {invitation.name}
                  </div>
                  <div className="mt-1 text-xs app-text-soft">
                    {titleCase(invitation.member?.role)} · {titleCase(invitation.subscription?.plan)}
                  </div>

                  <button
                    type="button"
                    onClick={() => acceptInvitation(invitation.id)}
                    disabled={busy === `accept:${invitation.id}`}
                    className="mt-4 inline-flex items-center gap-2 rounded-2xl bg-[var(--app-button-bg)] px-4 py-2 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    {busy === `accept:${invitation.id}`
                      ? t.accepting
                      : t.acceptInvitation}
                  </button>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {!selectedOrganization ? (
          <section className="rounded-3xl border app-surface-strong p-8">
            <h2 className="text-xl font-semibold app-text">{t.noTeam}</h2>
          </section>
        ) : (
          <>
            {organizations.length > 1 ? (
              <select
                value={selectedOrganization.id}
                onChange={changeOrganization}
                className="w-full rounded-2xl border px-4 py-3 text-sm"
              >
                {organizations.map((org) => (
                  <option key={org.id} value={org.id}>
                    {org.name}
                  </option>
                ))}
              </select>
            ) : null}

            <section className="grid gap-4 md:grid-cols-4">
              <div className="rounded-3xl border app-surface-strong p-5">
                <div className="text-xs uppercase tracking-[0.12em] app-text-soft">
                  {t.organization}
                </div>
                <div className="mt-2 truncate text-lg font-semibold app-text">
                  {selectedOrganization.name}
                </div>
              </div>
              <div className="rounded-3xl border app-surface-strong p-5">
                <div className="text-xs uppercase tracking-[0.12em] app-text-soft">
                  {t.plan}
                </div>
                <div className="mt-2 text-lg font-semibold app-text">
                  {titleCase(subscription?.plan)}
                </div>
              </div>
              <div className="rounded-3xl border app-surface-strong p-5">
                <div className="text-xs uppercase tracking-[0.12em] app-text-soft">
                  {t.seats}
                </div>
                <div className="mt-2 text-lg font-semibold app-text">
                  {seatsUsed} / {maxSeats ?? "—"}
                </div>
                {seatsAreFull ? (
                  <div className="mt-2 inline-flex rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-semibold text-amber-200">
                    {t.upgradeRequired}
                  </div>
                ) : null}
              </div>
              <div className="rounded-3xl border app-surface-strong p-5">
                <div className="text-xs uppercase tracking-[0.12em] app-text-soft">
                  {t.role}
                </div>
                <div className="mt-2 text-lg font-semibold app-text">
                  {titleCase(selectedOrganization.member?.role)}
                </div>
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
              <div className="rounded-3xl border app-surface-strong p-5">
                <h2 className="mb-4 text-xl font-semibold app-text">
                  {t.members}
                </h2>
                <div className="space-y-3">
                  {members.length ? (
                    members.map((member) => (
                      <div
                        key={member.user_id}
                        className="flex flex-col gap-3 rounded-2xl border app-surface p-4 md:flex-row md:items-center md:justify-between"
                      >
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold app-text">
                            {member.email || member.user_id}
                          </div>
                          <div className="text-xs app-text-soft">
                            {titleCase(member.role)} · {titleCase(member.status)}
                          </div>
                        </div>

                        {canManage ? (
                          <div className="flex flex-wrap gap-2">
                            <select
                              value={member.role}
                              disabled={
                                member.role === "owner" ||
                                busy === `role:${member.user_id}`
                              }
                              onChange={(event) =>
                                updateRole(member, event.target.value)
                              }
                              className="rounded-xl border px-3 py-2 text-xs"
                            >
                              <option value="member">{t.member}</option>
                              <option value="admin">{t.admin}</option>
                              <option value="owner">Owner</option>
                            </select>

                            <button
                              type="button"
                              disabled={
                                member.role === "owner" ||
                                busy === `remove:${member.user_id}`
                              }
                              onClick={() => removeMember(member)}
                              className="inline-flex items-center gap-2 rounded-xl border border-red-400/30 px-3 py-2 text-xs font-semibold text-red-200 transition hover:bg-red-400/10 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                              {t.remove}
                            </button>
                          </div>
                        ) : null}
                      </div>
                    ))
                  ) : (
                    <p className="rounded-2xl border app-surface p-4 text-sm app-text-muted">
                      {t.none}
                    </p>
                  )}
                </div>
                <p className="mt-4 text-xs app-text-soft">{t.ownerNote}</p>
              </div>

              <aside className="space-y-6">
                <form
                  onSubmit={inviteMember}
                  className="rounded-3xl border app-surface-strong p-5"
                >
                  <h2 className="text-xl font-semibold app-text">{t.invite}</h2>
                  {seatsAreFull ? (
                    <div className="mt-4 rounded-2xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-200">
                      <div className="font-semibold">{t.seatLimitReached}</div>
                      <div className="mt-1 text-xs">{t.upgradeRequiredDescription}</div>
                    </div>
                  ) : null}
                  <div className="mt-4 space-y-3">
                    <input
                      type="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      placeholder={t.email}
                      disabled={!canInvite || busy === "invite"}
                      className="w-full rounded-2xl border px-4 py-3 text-sm"
                    />
                    <select
                      value={role}
                      onChange={(event) => setRole(event.target.value)}
                      disabled={!canInvite || busy === "invite"}
                      className="w-full rounded-2xl border px-4 py-3 text-sm"
                    >
                      <option value="member">{t.member}</option>
                      <option value="admin">{t.admin}</option>
                    </select>
                    <button
                      type="submit"
                      disabled={!canInvite || !email.trim() || busy === "invite"}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-[var(--app-button-bg)] px-4 py-3 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Send className="h-4 w-4" />
                      {seatsAreFull ? t.upgradeRequired : t.send}
                    </button>
                  </div>
                </form>

                <div className="rounded-3xl border app-surface-strong p-5">
                  <h2 className="text-xl font-semibold app-text">
                    {t.invitations}
                  </h2>
                  <div className="mt-4 space-y-3">
                    {pendingMemberInvitations.length ? (
                      pendingMemberInvitations.map((member) => (
                        <div
                          key={member.user_id}
                          className="rounded-2xl border app-surface p-4"
                        >
                          <div className="truncate text-sm font-semibold app-text">
                            {member.email || member.user_id}
                          </div>
                          <div className="mt-1 text-xs app-text-soft">
                            {titleCase(member.role)} · {titleCase(member.status)}
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm app-text-muted">{t.none}</p>
                    )}
                  </div>
                </div>
              </aside>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
