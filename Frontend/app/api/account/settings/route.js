import { NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";

export async function PATCH(req) {
  let accessToken = "";

  try {
    const session = await auth0.getSession();

    if (!session) {
      return NextResponse.json(
        {
          detail: {
            error: "authorization_required",
            message: "You must be signed in.",
          },
        },
        { status: 401 },
      );
    }

    const tokenSet = await auth0.getAccessToken();
    accessToken =
      typeof tokenSet === "string" ? tokenSet : tokenSet?.token || "";
  } catch {
    return NextResponse.json(
      {
        detail: {
          error: "authorization_required",
          message: "Could not load session.",
        },
      },
      { status: 401 },
    );
  }

  const body = await req.json();

  const backendRes = await fetch(
    `${process.env.BACKEND_URL}/api/v1/account/settings`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(body),
      cache: "no-store",
    },
  );

  const contentType = backendRes.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await backendRes.json()
    : { detail: { message: await backendRes.text() } };

  return NextResponse.json(data, { status: backendRes.status });
}
