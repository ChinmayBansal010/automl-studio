import { ArrowLeft, ArrowRight, CheckCircle2, CircleHelp, Home, RefreshCw, ShieldAlert, WifiOff } from 'lucide-react'

const pages: Record<number, { eyebrow: string; title: string; description: string; detail: string; action: string; icon: typeof ShieldAlert }> = {
  400: { eyebrow: 'REQUEST ERROR', title: 'That request could not be understood.', description: 'The request was incomplete or contained information the service could not process.', detail: 'Check the information and try again. If the problem continues, return to your workspace and repeat the action.', action: 'Return to workspace', icon: CircleHelp },
  401: { eyebrow: 'SIGN-IN REQUIRED', title: 'Your session needs attention.', description: 'You may have been signed out or your session may have expired.', detail: 'Sign in again to continue. Your locally saved workspace data is not cleared by this page.', action: 'Sign in', icon: ShieldAlert },
  403: { eyebrow: 'ACCESS DENIED', title: 'You do not have access to this page.', description: 'The account currently signed in is not permitted to open this resource.', detail: 'Check that you are using the correct account. If you believe access should be available, contact the project maintainer.', action: 'Return to workspace', icon: ShieldAlert },
  404: { eyebrow: 'PAGE NOT FOUND', title: 'This page is not here.', description: 'The address may be incorrect, or the page may have moved.', detail: 'Check the URL or return to the workspace and continue from the navigation.', action: 'Go to home', icon: CircleHelp },
  405: { eyebrow: 'METHOD NOT ALLOWED', title: 'That action is not supported here.', description: 'The request used a method this resource does not accept.', detail: 'Return to the application and use the available controls for this operation.', action: 'Return to workspace', icon: ShieldAlert },
  409: { eyebrow: 'CONFLICT', title: 'The resource has changed.', description: 'The action conflicts with the current state of this resource.', detail: 'Refresh the page, review the latest dataset or experiment state, then retry if the action is still needed.', action: 'Refresh page', icon: RefreshCw },
  413: { eyebrow: 'FILE TOO LARGE', title: 'This file is larger than the upload limit.', description: 'The server rejected the request because its payload is too large.', detail: 'Choose a smaller CSV file or reduce the amount of data before uploading.', action: 'Choose another file', icon: CircleHelp },
  422: { eyebrow: 'INVALID DATA', title: 'The submitted data could not be processed.', description: 'The request was understood, but one or more values failed validation.', detail: 'Review the selected columns, target and input values before trying again.', action: 'Return to workspace', icon: CircleHelp },
  408: { eyebrow: 'REQUEST TIMEOUT', title: 'The request took too long.', description: 'The service did not finish responding in time.', detail: 'Check your connection and retry. Large datasets and model-training jobs can take longer than ordinary page requests.', action: 'Try again', icon: RefreshCw },
  410: { eyebrow: 'RESOURCE REMOVED', title: 'This resource is no longer available.', description: 'The requested item has been removed or is no longer retained.', detail: 'Return to your workspace to select an available dataset or experiment.', action: 'Go to home', icon: CircleHelp },
  429: { eyebrow: 'TOO MANY REQUESTS', title: 'Give the service a moment.', description: 'Too many requests were received in a short period.', detail: 'Wait briefly before repeating the action. Repeated clicks will not make the queue move faster.', action: 'Try again', icon: RefreshCw },
  500: { eyebrow: 'SERVER ERROR', title: 'Something failed on our side.', description: 'The application encountered an unexpected error.', detail: 'Your request may not have completed. Return to the workspace and check the result before submitting it again.', action: 'Return to workspace', icon: ShieldAlert },
  501: { eyebrow: 'NOT IMPLEMENTED', title: 'This operation is not available.', description: 'The server does not currently support the requested operation.', detail: 'Choose another available operation or check the project issue tracker for implementation updates.', action: 'Return to workspace', icon: CircleHelp },
  502: { eyebrow: 'BAD GATEWAY', title: 'The service returned an invalid response.', description: 'A gateway could not get a valid response from the application server.', detail: 'This is usually temporary. Wait a moment and try again.', action: 'Retry', icon: RefreshCw },
  503: { eyebrow: 'SERVICE UNAVAILABLE', title: 'The service is taking a break.', description: 'AutoML Studio or one of its supporting services is temporarily unavailable.', detail: 'This can happen during maintenance or when the backend is waking up. Your browser may still have locally backed-up datasets.', action: 'Check again', icon: WifiOff },
  504: { eyebrow: 'GATEWAY TIMEOUT', title: 'The service did not respond in time.', description: 'The upstream server took too long to complete the request.', detail: 'Check the service status and retry. Avoid submitting the same training job repeatedly.', action: 'Retry', icon: RefreshCw },
}

export function SystemPage({ code, onHome, onLogin }: { code: number; onHome: () => void; onLogin: () => void }) {
  const page = pages[code] || pages[404]
  const Icon = page.icon
  const needsLogin = code === 401
  const canRetry = [408, 429, 502, 503, 504].includes(code)
  return (
    <main className="system-page">
      <header className="system-topbar"><a href="/" className="system-brand"><span className="system-brand-mark">A</span><strong>AutoML Studio</strong></a><span>APPLICATION STATUS</span></header>
      <section className="system-content">
        <div className="system-code" aria-label={'HTTP ' + code}>{String(code).split('').map((digit, index) => <span key={index}>{digit}</span>)}</div>
        <div className="system-copy">
          <span className="system-eyebrow"><Icon size={14}/>{page.eyebrow}</span>
          <h1>{page.title}</h1>
          <p className="system-description">{page.description}</p>
          <p className="system-detail">{page.detail}</p>
          <div className="system-actions">
            <button className="primary" type="button" onClick={() => canRetry ? window.location.reload() : needsLogin ? onLogin() : onHome()}>
              {canRetry ? <RefreshCw size={16}/> : needsLogin ? <ArrowRight size={16}/> : <Home size={16}/>}
              {page.action}
            </button>
            <button className="secondary" type="button" onClick={() => window.history.length > 1 ? window.history.back() : onHome()}><ArrowLeft size={15}/> Go back</button>
          </div>
          <div className="system-reference"><CheckCircle2 size={15}/> No action is taken automatically. You remain in control of retrying.</div>
        </div>
      </section>
      <footer className="system-footer"><span>AutoML Studio</span><a href="https://github.com/ChinmayBansal010/automl-studio/issues" target="_blank" rel="noreferrer">Report a problem <ArrowRight size={12}/></a></footer>
    </main>
  )
}
