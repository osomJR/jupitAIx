export default function VerifyEmailRequiredPage() {
  return (
    <main style={{ padding: 24 }}>
      <h1>Verify your email</h1>
      <p>
        We sent you a verification link. Please open your email, click the link,
        then sign in again.
      </p>
      <a href="/auth/login?returnTo=/">Sign in again</a>
    </main>
  );
}
