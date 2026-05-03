export default function VerifyEmailRequiredPage() {
  return (
    <main style={{ padding: 24 }}>
      <h1>Verify your email</h1>
      <p>
        We sent you a verification link. Please open and verify your email and
        sign in again.
      </p>

      <a
        href="/auth/login?returnTo=/"
        style={{
          display: "inline-block",
          marginTop: 16,
          padding: "10px 16px",
          backgroundColor: "#2563eb",
          color: "white",
          textDecoration: "none",
          borderRadius: 6,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        Sign in again
      </a>
    </main>
  );
}