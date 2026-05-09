import { NextResponse } from "next/server";
import { auth0 } from "@/lib/auth0";

export async function POST(req) {
  const incomingFormData = await req.formData();
  const outboundFormData = new FormData();

  for (const [key, value] of incomingFormData.entries()) {
    if (
      typeof value === "object" &&
      value !== null &&
      typeof value.arrayBuffer === "function" &&
      typeof value.name === "string"
    ) {
      const buffer = await value.arrayBuffer();

      const fileBlob = new Blob([buffer], {
        type: value.type || "application/octet-stream",
      });

      outboundFormData.append(key, fileBlob, value.name);
    } else {
      outboundFormData.append(key, value);
    }
  }

  let accessToken = "";

  try {
    const session = await auth0.getSession();

    if (session) {
      const tokenSet = await auth0.getAccessToken();
      accessToken =
        typeof tokenSet === "string" ? tokenSet : tokenSet?.token || "";
    }
  } catch {
    accessToken = "";
  }

  const headers = {};

  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }

  const backendRes = await fetch(
    `${process.env.BACKEND_URL}/api/v1/analyzer/compliance/preview`,
    {
      method: "POST",
      headers,
      body: outboundFormData,
    },
  );

  const contentType = backendRes.headers.get("content-type") || "";

  const data = contentType.includes("application/json")
    ? await backendRes.json()
    : { detail: { message: await backendRes.text() } };

  return NextResponse.json(data, { status: backendRes.status });
}
