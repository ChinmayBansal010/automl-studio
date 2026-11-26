import { useEffect } from 'react'
import { AutoMLMark } from './components/AutoMLMark'
import { Activity, ArrowRight, BarChart3, BrainCircuit, CheckCircle2, Database, FileSpreadsheet, FlaskConical, Github, Gauge, RefreshCw, ShieldCheck, Target } from 'lucide-react'

type Props = { page: 'home' | 'about' | 'privacy' | 'terms' | 'contact'; onLogin: () => void; onSignup: () => void }
const updated = '24 November 2026'
const descriptions: Record<Props['page'], string> = {
  home: 'AutoML Studio is a practical machine-learning workspace for inspecting datasets, preparing features, comparing models and generating predictions.',
  about: 'Learn about AutoML Studio, its workflow, supported machine-learning tasks and project maintainer.',
  privacy: 'Read how AutoML Studio handles account information, uploaded datasets and browser-side workspace backups.',
  terms: 'Terms for using AutoML Studio, including dataset ownership, model outputs and service availability.',
  contact: 'Contact the AutoML Studio maintainer, report a defect or request a feature through the project issue tracker.',
}
const titles: Record<Props['page'], string> = { home: 'AutoML Studio | Practical machine-learning workspace', about: 'About AutoML Studio', privacy: 'Privacy | AutoML Studio', terms: 'Terms of use | AutoML Studio', contact: 'Contact | AutoML Studio' }

export function PublicPage({ page, onLogin, onSignup }: Props) {
  useEffect(() => {
    document.title = titles[page]
    let description = document.querySelector('meta[name="description"]')
    if (!description) { description = document.createElement('meta'); description.setAttribute('name', 'description'); document.head.appendChild(description) }
    description.setAttribute('content', descriptions[page])
    let canonical = document.querySelector('link[rel="canonical"]') as HTMLLinkElement | null
    if (!canonical) { canonical = document.createElement('link'); canonical.rel = 'canonical'; document.head.appendChild(canonical) }
    canonical.href = window.location.origin + (page === 'home' ? '/' : '/' + page)
    const setMeta = (selector: string, attribute: string, value: string) => {
      let element = document.querySelector(selector) as HTMLMetaElement | null
      if (!element) { element = document.createElement('meta'); const [key, name] = selector.match(/\[(name|property)="([^"]+)"\]/)?.slice(1) || []; if (key && name) element.setAttribute(key, name); document.head.appendChild(element) }
      element.setAttribute(attribute, value)
    }
    setMeta('meta[property="og:title"]', 'content', titles[page])
    setMeta('meta[property="og:description"]', 'content', descriptions[page])
    setMeta('meta[property="og:url"]', 'content', canonical.href)
    setMeta('meta[name="twitter:card"]', 'content', 'summary')
    let schema = document.getElementById('automl-software-schema') as HTMLScriptElement | null
    if (!schema) { schema = document.createElement('script'); schema.id = 'automl-software-schema'; schema.type = 'application/ld+json'; document.head.appendChild(schema) }
    schema.textContent = JSON.stringify({ '@context': 'https://schema.org', '@type': 'SoftwareApplication', name: 'AutoML Studio', applicationCategory: 'DeveloperApplication', operatingSystem: 'Web', description: descriptions[page], url: window.location.origin, author: { '@type': 'Person', name: 'Chinmay Bansal' } })
  }, [page])

  const nav = <nav className="public-nav" aria-label="Main navigation"><a className="public-brand" href="/"><span className="public-brand-mark"><AutoMLMark /></span>AutoML Studio</a><div><a href="/about">About</a><a href="/privacy">Privacy</a><a href="/terms">Terms</a><a href="/contact">Contact</a></div><button className="secondary" type="button" onClick={onLogin}>Sign in</button></nav>
  if (page === 'home') return (
    <main className="public-page">
      {nav}
      <section className="public-hero">
        <div className="public-hero-copy">
          <span className="public-kicker">THE PRACTICAL MACHINE-LEARNING WORKSPACE</span>
          <h1>From raw data to <em>reliable model decisions.</em></h1>
          <p>Prepare datasets, explore patterns, train multiple algorithms and evaluate results in one structured workspace. Keep every experiment connected to the data and configuration that produced it.</p>
          <div className="public-hero-actions"><button className="primary" onClick={onSignup}>Create your workspace <ArrowRight size={16}/></button><button className="secondary" onClick={onLogin}>Sign in</button><a className="public-tour-link" href="#workflow">Explore workflow <ArrowRight size={15}/></a></div>
          <div className="public-hero-proof"><span><CheckCircle2 size={15}/> Classification and regression</span><span><CheckCircle2 size={15}/> Reproducible experiment history</span><span><CheckCircle2 size={15}/> CSV-first workflow</span></div>
        </div>
        <div className="public-product-sheet">
          <div className="sheet-top"><span>WORKSPACE / DATA OVERVIEW</span><span className="sheet-status">● EXAMPLE WORKSPACE</span></div>
          <div className="sheet-file"><FileSpreadsheet size={19}/><div><strong>customer_churn.csv</strong><small>12,480 rows · 18 columns</small></div></div>
          <div className="sheet-stats"><div><small>ROWS</small><strong>12,480</strong></div><div><small>COLUMNS</small><strong>18</strong></div><div><small>MISSING</small><strong>2.4%</strong></div></div>
          <div className="sheet-analysis"><div><span>DATA QUALITY</span><strong>97.6%</strong></div><div className="sheet-quality-track"><i/></div><div className="sheet-chart"><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/></div><div className="sheet-model-row"><span><BrainCircuit size={15}/> Random Forest</span><strong>Model evaluated</strong></div><div className="sheet-metric-row"><div><small>ACCURACY</small><strong>92.4%</strong></div><div><small>F1 SCORE</small><strong>0.91</strong></div><div><small>ROC AUC</small><strong>0.95</strong></div></div></div>
          <div className="sheet-footer">Illustrative product preview · sample values, not live results</div>
        </div>
      </section>
      <section className="public-trust-strip" aria-label="Platform principles"><div><strong>One connected workflow</strong><span>Move from inspection through prediction without losing context.</span></div><div><strong>Transparent evaluation</strong><span>Compare metrics and inspect model performance.</span></div><div><strong>Your choices stay explicit</strong><span>Choose transformations, targets and algorithms.</span></div></section>
      <section className="public-workflow">
        <div><span className="public-kicker">A CLEAR, REPEATABLE PROCESS</span><h2>Every stage has a purpose.</h2><p>Work through the analysis in order, or return to any stage when you need to refine the dataset or compare a different approach.</p></div>
        <div className="public-feature-list">
          <article><Database/><span>01 / PROFILE</span><h3>Understand the dataset</h3><p>Inspect columns, data types, summary statistics, missing values and data quality before making changes.</p></article>
          <article><FileSpreadsheet/><span>02 / PREPARE</span><h3>Prepare features</h3><p>Handle missing values, remove or clip outliers, convert columns and preserve dataset versions.</p></article>
          <article><BarChart3/><span>03 / EXPLORE</span><h3>Explore patterns</h3><p>Build visualizations, compare distributions and examine relationships between variables.</p></article>
          <article><FlaskConical/><span>04 / TRAIN</span><h3>Train and compare</h3><p>Run classification or regression algorithms, configure train-test splits and compare candidates.</p></article>
          <article><Gauge/><span>05 / EVALUATE</span><h3>Measure performance</h3><p>Review evaluation metrics, confusion matrices and prediction behaviour before selecting a model.</p></article>
          <article><Target/><span>06 / PREDICT</span><h3>Use the trained model</h3><p>Generate predictions for new CSV files and export the results for downstream work.</p></article>
        </div>
      </section>
      <section className="public-capabilities">
        <div className="public-capabilities-heading"><span className="public-kicker">BUILT FOR HANDS-ON ML</span><h2>Useful tools around the entire experiment.</h2><p>Not just a model-training button. Keep the work organized before, during and after a run.</p></div>
        <div className="public-capability-grid">
          <article><Activity/><h3>Data quality and profiling</h3><p>Review dataset structure and identify issues before training.</p></article>
          <article><RefreshCw/><h3>Preprocessing and feature work</h3><p>Apply transformations with a visible, deliberate workflow.</p></article>
          <article><BarChart3/><h3>Interactive visual analysis</h3><p>Build charts to inspect distributions and relationships.</p></article>
          <article><BrainCircuit/><h3>Model comparison</h3><p>Compare supported algorithms using task-appropriate metrics.</p></article>
          <article><Activity/><h3>Experiment history</h3><p>Keep a record of training runs and revisit saved results.</p></article>
          <article><ShieldCheck/><h3>Workspace separation</h3><p>Access datasets and experiments through your authenticated account.</p></article>
        </div>
      </section>
      <section className="public-bottom-cta"><div><span className="public-kicker">START WITH A CSV</span><h2>Make your next machine-learning experiment easier to inspect and improve.</h2><p>Create a workspace, upload a dataset and take the first step.</p></div><button className="primary" onClick={onSignup}>Create your workspace <ArrowRight size={16}/></button></section>
      <PublicFooter/>
    </main>
  )
  const content = {
    about: <><span className="public-kicker">PROJECT OVERVIEW</span><h1>About AutoML Studio</h1><p className="public-lead">A focused workspace for the practical parts of an applied machine-learning project.</p><h2>What it does</h2><p>AutoML Studio brings dataset inspection, preprocessing, feature engineering, visual analysis, model training, evaluation, experiment history and prediction into one application. It supports supervised classification and regression workflows.</p><h2>How it works</h2><p>The workflow is explicit: upload a CSV, inspect the profile, choose preprocessing operations, select a target and algorithms, then review evaluation results. The application uses deterministic data-processing and machine-learning libraries. It does not require a generative assistant to complete these tasks.</p><h2>Project maintainer</h2><p><strong>Chinmay Bansal</strong><br/>Project maintainer and developer.</p><p className="public-updated">Last updated: {updated}</p></>,
    privacy: <><span className="public-kicker">DATA HANDLING</span><h1>Privacy notice</h1><p className="public-lead">Last updated {updated}. This notice describes the main data flows in AutoML Studio.</p><h2>Account information</h2><p>Signing in uses the configured authentication service. Your email address and authentication records are handled by that service to create and maintain your account.</p><h2>Datasets and model operations</h2><p>When you upload a CSV or request a data or model operation, the file and relevant inputs are sent to the application backend to perform that operation. Do not upload personal, confidential or regulated data unless you have the authority to do so and have reviewed the deployment's data-handling arrangements.</p><h2>Browser-side backups</h2><p>The application may keep dataset backups, experiment history and dashboard preferences in your browser's local storage mechanisms, including IndexedDB and localStorage. These records are tied to the browser profile and are not a substitute for a separate backup.</p><h2>Retention and deletion</h2><p>Retention of server-side datasets, logs and model artifacts depends on the backend deployment and its storage configuration. This page does not promise a retention period. Clear local browser data to remove browser-side copies; contact the project maintainer for questions about server-side records.</p><h2>Changes</h2><p>This notice may be updated when application data flows change. The date above identifies the latest stated revision.</p></>,
    terms: <><span className="public-kicker">USE OF THE APPLICATION</span><h1>Terms of use</h1><p className="public-lead">Last updated {updated}. By using AutoML Studio, you agree to use it responsibly.</p><h2>Your data and permissions</h2><p>You must own the data you upload or have permission to process it. You are responsible for removing information you are not authorized to use and for complying with applicable laws and institutional requirements.</p><h2>Model outputs</h2><p>Training metrics, predictions and exported artifacts are computational outputs, not guarantees of accuracy or suitability. Validate models independently before using them in operational, financial, medical, legal or other consequential decisions.</p><h2>Acceptable use</h2><p>Do not attempt to disrupt the service, bypass authentication, access another user's information or upload malicious files. Do not use the application to violate another person's rights.</p><h2>Availability</h2><p>The application and its supporting services may be unavailable, changed or discontinued. Keep independent copies of important datasets, experiments and model artifacts.</p><h2>Updates</h2><p>These terms may change as the project evolves. Continued use after a revision means you accept the updated terms.</p></>,
    contact: <><span className="public-kicker">HUMAN SUPPORT</span><h1>Contact the maintainer</h1><p className="public-lead">Found a defect, have a reproducible failure or want to propose a feature? Open an issue in the project repository.</p><a className="public-issue-link" href="https://github.com/ChinmayBansal010/automl-studio/issues/new/choose" target="_blank" rel="noreferrer"><Github size={18}/><span><strong>Open a GitHub issue</strong><small>Include steps to reproduce, expected behaviour and relevant logs. Never include passwords, tokens or private dataset contents.</small></span><ArrowRight size={17}/></a><p className="public-updated">Project maintainer: Chinmay Bansal</p></>,
  }[page]
  return <main className="public-page">{nav}<article className="public-document">{content}</article><PublicFooter/></main>
}

function PublicFooter() {
  return <footer className="public-footer"><span>© {new Date().getFullYear()} AutoML Studio · Maintained by Chinmay Bansal</span><div><a href="https://github.com/ChinmayBansal010/automl-studio" target="_blank" rel="noreferrer">Source code</a><a href="/privacy">Privacy</a><a href="/terms">Terms</a><a href="/contact">Contact</a></div></footer>
}
