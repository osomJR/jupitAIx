import { NextResponse } from "next/server";

const ALLOWED_FEATURES = [
  "convert",
  "summarize",
  "grammar-correct",
  "translate",
  "transcribe",
  "explain",
  "generate-questions",
  "generate-answers",
];

export async function GET() {
  return NextResponse.json({
    ok: true,
    message: "Analyzer proxy route is working. Send a POST request.",
  });
}

export async function POST(req, context) {
  const { feature } = await context.params;

  if (!ALLOWED_FEATURES.includes(feature)) {
    return NextResponse.json(
      {
        detail: {
          error: "invalid_feature",
          message: "Unsupported analyzer feature.",
        },
      },
      { status: 400 },
    );
  }

  const formData = await req.formData();

  const backendRes = await fetch(
    `${process.env.BACKEND_URL}/api/v1/analyzer/${feature}`,
    {
      method: "POST",
      headers: {
        Authorization: req.headers.get("authorization") || "",
      },
      body: formData,
    },
  );

  const contentType = backendRes.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await backendRes.json()
    : { detail: { message: await backendRes.text() } };

  return NextResponse.json(data, { status: backendRes.status });
}
