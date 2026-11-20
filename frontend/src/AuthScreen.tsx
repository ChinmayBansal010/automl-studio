import { useState, type FormEvent } from 'react'
import { LockKeyhole, Mail, Sparkles } from 'lucide-react'
import { authClient } from './auth'

type AuthUser = { id: string; email: string; name?: string }

async function authRequest(path: string, body: Record<string, string>) {
  const response = await fetch(`/auth/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.message || result.error?.message || `Authentication request failed (${response.status}).`)
  return result
}

export function AuthScreen({ onAuthenticated }: { onAuthenticated: (user: AuthUser) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [otp, setOtp] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [needsVerification, setNeedsVerification] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function finishSignIn() {
    let result = await authClient.getSession()
    if (result.error || !result.data?.user || !result.data.session?.token) {
      result = await authClient.getSession({
        fetchOptions: { headers: { 'X-Force-Fetch': 'true' } },
      })
    }
    if (result.error) throw new Error(result.error.message)
    if (!result.data?.user || !result.data.session?.token) {
      throw new Error('No active session was returned. Please sign in again.')
    }
    onAuthenticated(result.data.user)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    setNotice('')
    try {
      if (needsVerification) {
        await authRequest('email-otp/verify-email', { email, otp })
        let session = await authClient.getSession()
        if (session.error || !session.data?.user || !session.data.session?.token) {
          session = await authClient.getSession({
            fetchOptions: { headers: { 'X-Force-Fetch': 'true' } },
          })
        }
        if (!session.error && session.data?.user && session.data.session?.token) {
          onAuthenticated(session.data.user)
        } else {
          const signIn = await authClient.signIn.email({ email, password })
          if (signIn.error) throw new Error(signIn.error.message)
          await finishSignIn()
        }
        return
      }

      const result = isSignUp
        ? await authClient.signUp.email({ name: email.split('@')[0] || 'User', email, password })
        : await authClient.signIn.email({ email, password })
      if (result.error) throw new Error(result.error.message)

      if (isSignUp) {
        setNeedsVerification(true)
        setNotice('Check your email for a verification code.')
      } else {
        await finishSignIn()
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  async function resendCode() {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await authRequest('email-otp/send-verification-otp', { email, type: 'email-verification' })
      setNotice('A new verification code was sent.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-page">
      <div className="auth-wrapper">
        <div className="auth-hero">
          <div className="auth-hero-icon"><Sparkles size={40} /></div>
          <h2>AutoML Studio</h2>
          <p>Your private machine-learning workspace. Upload CSV data, preprocess, visualize, train models, and generate predictions — all in one place.</p>
          <ul className="auth-features">
            <li>Automated model comparison & leaderboard</li>
            <li>Interactive EDA & chart builder</li>
            <li>One-click preprocessing & feature engineering</li>
            <li>Download trained models & predictions</li>
          </ul>
        </div>
        <section className="auth-card">
          <div className="auth-brand"><Sparkles size={24} /><span>AutoML Studio</span></div>
          <h1>{needsVerification ? 'Verify your email' : isSignUp ? 'Create your account' : 'Welcome back'}</h1>
          <p>{needsVerification ? `Enter the code sent to ${email}.` : 'Sign in to your private machine-learning workspace.'}</p>
          <form onSubmit={submit}>
            {!needsVerification && <>
              <label className="auth-field"><span>Email</span><span className="auth-input"><Mail size={17} /><input type="email" autoComplete="email" required value={email} onChange={event => setEmail(event.target.value)} placeholder="you@example.com" /></span></label>
              <label className="auth-field"><span>Password</span><span className="auth-input"><LockKeyhole size={17} /><input type="password" autoComplete={isSignUp ? 'new-password' : 'current-password'} minLength={8} required value={password} onChange={event => setPassword(event.target.value)} placeholder="Min. 8 characters" /></span></label>
            </>}
            {needsVerification && <label className="auth-field"><span>Verification code</span><input className="auth-otp" inputMode="numeric" autoComplete="one-time-code" required value={otp} onChange={event => setOtp(event.target.value)} placeholder="Enter 6-digit code" /></label>}
            {error && <div className="auth-error" role="alert">{error}</div>}
            {notice && <div className="auth-notice" role="status">{notice}</div>}
            <button className="primary auth-submit" disabled={busy}>{busy ? 'Please wait…' : needsVerification ? 'Verify and continue' : isSignUp ? 'Create account' : 'Sign in'}</button>
          </form>
          {needsVerification ? <button className="auth-link" disabled={busy} onClick={resendCode}>Resend verification code</button> : <button className="auth-link" onClick={() => { setIsSignUp(value => !value); setError('') }}>{isSignUp ? 'Already have an account? Sign in' : 'New here? Create an account'}</button>}
        </section>
      </div>
    </main>
  )
}
