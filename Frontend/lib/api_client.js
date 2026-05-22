export async function getAccessToken() {
  const res = await fetch("/auth/access-token", {
    method: "GET",
    credentials: "include",
  });

  if (!res.ok) {
    throw new Error("Could not get access token");
  }

  const data = await res.json();
  return data.token;
}

export async function postAnalyzerFeature(
  feature,
  formData,
  requiresAuth = false,
) {
  const headers = {};

  if (requiresAuth) {
    const token = await getAccessToken();
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`/api/analyzer/${feature}`, {
    method: "POST",
    headers,
    body: formData,
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(getErrorMessage(data));
  }

  return data;
}

function getErrorMessage(data, fallback = "Request failed") {
  return (
    data?.error?.message ||
    data?.detail?.message ||
    data?.detail?.error ||
    data?.message ||
    fallback
  );
}

function normalizeAccountMe(data) {
  if (data?.user && data?.entitlement) {
    return data;
  }

  // Defensive fallback for older Auth0 profile-shaped responses.
  if (data && (data.sub || data.id || data.email || data.name || data.picture)) {
    return {
      user: {
        id: data.id || data.sub || null,
        name: data.name || data.nickname || null,
        email: data.email || null,
        picture: data.picture || null,
      },
      settings: data.settings || null,
      entitlement: data.entitlement || null,
    };
  }

  return data;
}

export async function getAccountMe() {
  const res = await fetch("/api/account/me", {
    method: "GET",
    credentials: "include",
    cache: "no-store",
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const error = new Error(getErrorMessage(data, "Could not load account."));
    error.status = res.status;
    error.payload = data;
    throw error;
  }

  return normalizeAccountMe(data);
}

