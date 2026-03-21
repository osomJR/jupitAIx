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
    throw new Error(
      data?.detail?.message || data?.detail?.error || "Request failed",
    );
  }

  return data;
}
