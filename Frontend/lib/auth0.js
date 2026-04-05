import { NextResponse } from "next/server";
import { Auth0Client } from "@auth0/nextjs-auth0/server";

const appBaseUrl = process.env.APP_BASE_URL;

export const auth0 = new Auth0Client({
  appBaseUrl,

  authorizationParameters: {
    redirect_uri: `${appBaseUrl}/auth/callback`,
    audience: process.env.AUTH0_AUDIENCE,
    scope: process.env.AUTH0_SCOPE,
  },

  async onCallback(error, context, session) {
    if (error) {
      console.error("Auth0 callback error:", {
        code: error.code,
        message: error.message,
        cause: error.cause?.message,
      });

      const details = [error.code, error.message, error.cause?.message]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      if (details.includes("email_not_verified")) {
        return NextResponse.redirect(
          new URL("/verify-email-required", appBaseUrl),
        );
      }

      const url = new URL("/auth-error", appBaseUrl);
      if (error.code) url.searchParams.set("code", error.code);
      if (error.message) url.searchParams.set("message", error.message);

      return NextResponse.redirect(url);
    }

    return NextResponse.redirect(new URL(context.returnTo || "/", appBaseUrl));
  },
});
