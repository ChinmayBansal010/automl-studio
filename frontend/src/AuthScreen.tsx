import { useEffect, useState, type FormEvent } from 'react'
import { ArrowRight, Check, LockKeyhole, Mail, ShieldCheck, Sparkles } from 'lucide-react'
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

export function AuthScreen({ onAuthenticated, initialMode = 'login', onModeChange }: { onAuthenticated: (user: AuthUser) => void; initialMode?: 'login' | 'signup'; onModeChange?: (mode: 'login' | 'signup') => void }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [otp, setOtp] = useState('')
  const [isSignUp, setIsSignUp] = useState(initialMode === 'signup')
  useEffect(() => setIsSignUp(initialMode === 'signup'), [initialMode])
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
        ? await authClient.signUp.email({ name: name.trim() || email.split('@')[0] || 'User', email, password })
        : await authClient.signIn.email({ email, password })
      if (result.error) throw new Error(result.error.message)

      if (isSignUp) {
        setNeedsVerification(true)
        setNotice('Check your inbox for the verification code.')
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

  function switchMode(signUp: boolean) {
    setIsSignUp(signUp)
    onModeChange?.(signUp ? 'signup' : 'login')
    setError('')
    setNotice('')
  }

  return (
    <main className="auth-page">
      <header className="auth-topbar">
        <a className="auth-topbrand" href="/" aria-label="AutoML Studio home">
          <span className="auth-topbrand-mark"><Sparkles size={19} /></span>
          <span>AutoML <strong>Studio</strong></span>
        </a>
        <div className="auth-topbar-meta"><ShieldCheck size={15} /> Secure workspace access</div>
      </header>

      <div className="auth-layout">
        <section className="auth-showcase">
          <div className="auth-eyebrow"><span /> YOUR DATA. YOUR MODELS. YOUR WORKSPACE.</div>
          <h1>Machine learning,<br /><span>from data to decisions.</span></h1>
          <p className="auth-showcase-copy">Inspect data, prepare features, compare models and generate predictions from one organized workspace.</p>

          <div className="auth-feature-list">
            <div><span className="auth-feature-icon"><Check size={15} /></span><span><strong>Understand the data</strong><small>Review data quality, types and distributions.</small></span></div>
            <div><span className="auth-feature-icon"><Check size={15} /></span><span><strong>Prepare and train</strong><small>Transform features and compare algorithms.</small></span></div>
            <div><span className="auth-feature-icon"><Check size={15} /></span><span><strong>Review results</strong><small>Keep experiments, models and predictions together.</small></span></div>
          </div>

          <div className="auth-showcase-note"><span className="auth-note-dot" /> A private workspace for your machine-learning workflow</div>
          <div className="auth-orbit auth-orbit-one" />
          <div className="auth-orbit auth-orbit-two" />
        </section>

        <section className="auth-panel">
          <div className="auth-card">
            <div className="auth-card-heading">
              <div className="auth-card-icon"><Sparkles size={21} /></div>
              <div><span className="auth-card-kicker">AUTOML STUDIO</span><h2>{needsVerification ? 'Verify your email' : isSignUp ? 'Create your account' : 'Welcome back'}</h2></div>
            </div>
            <p className="auth-card-description">
              {needsVerification
                ? `Enter the verification code sent to ${email}.`
                : isSignUp
                  ? 'Create an account to set up your private workspace.'
                  : 'Sign in to continue to your machine-learning workspace.'}
            </p>

            {!needsVerification && (
              <div className="auth-mode-switch" role="tablist" aria-label="Authentication mode">
                <button type="button" role="tab" aria-selected={!isSignUp} className={!isSignUp ? 'selected' : ''} onClick={() => switchMode(false)}>Sign in</button>
                <button type="button" role="tab" aria-selected={isSignUp} className={isSignUp ? 'selected' : ''} onClick={() => switchMode(true)}>Create account</button>
              </div>
            )}

            <form onSubmit={submit}>
              {!needsVerification && <>
                {isSignUp && <label className="auth-field"><span>Full name</span><span className="auth-input"><input type="text" autoComplete="name" required value={name} onChange={event => setName(event.target.value)} placeholder="Your name" /></span></label>}
                <label className="auth-field"><span>Email address</span><span className="auth-input"><Mail size={17} /><input type="email" autoComplete="email" required value={email} onChange={event => setEmail(event.target.value)} placeholder="name@example.com" /></span></label>
                <label className="auth-field"><span>Password</span><span className="auth-input"><LockKeyhole size={17} /><input type="password" autoComplete={isSignUp ? 'new-password' : 'current-password'} minLength={8} required value={password} onChange={event => setPassword(event.target.value)} placeholder="At least 8 characters" /></span></label>
              </>}
              {needsVerification && <label className="auth-field"><span>Verification code</span><input className="auth-otp" inputMode="numeric" autoComplete="one-time-code" required value={otp} onChange={event => setOtp(event.target.value)} placeholder="Enter your code" /></label>}
              {error && <div className="auth-error" role="alert">{error}</div>}
              {notice && <div className="auth-notice" role="status">{notice}</div>}
              <button className="primary auth-submit" disabled={busy}>
                {busy ? 'Please wait…' : needsVerification ? 'Verify and continue' : isSignUp ? 'Create account' : 'Sign in'}
                {!busy && <ArrowRight size={16} />}
              </button>
            </form>

            {needsVerification
              ? <button className="auth-link" disabled={busy} onClick={resendCode}>Resend verification code</button>
              : <p className="auth-switch-copy">{isSignUp ? 'Already have an account?' : 'New to AutoML Studio?'} <button className="auth-link-inline" type="button" onClick={() => switchMode(!isSignUp)}>{isSignUp ? 'Sign in' : 'Create an account'}</button></p>}
            <div className="auth-security-note"><ShieldCheck size={14} /> Your account is protected by secure authentication.</div>
          </div>
          <div className="auth-panel-footer">AutoML Studio <span>·</span> Machine learning, organized.</div>
        </section>
      </div>
    </main>
  )
}
