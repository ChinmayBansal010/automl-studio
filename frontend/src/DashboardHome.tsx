import { useRef } from 'react'
import { ArrowRight, Clock3, Database, FilePlus2, FlaskConical, FolderOpen } from 'lucide-react'

type Dataset = { dataset_id: string; filename: string; saved_at?: string }
type Job = { job_id: string; best_model?: string; model_name?: string; task?: string; target?: string; status?: string; created_at?: string; model_id?: string }
type ActiveDataset = { dataset_id: string; filename: string; rows?: number; columns?: number; missing_total?: number } | null

type Props = {
  activeDataset: ActiveDataset
  datasets: Dataset[]
  jobs: Job[]
  loading: boolean
  onUpload: (file?: File) => void
  onOpenDataset: (id: string) => void
  onOpenOverview: () => void
  onOpenHistory: () => void
}

export function DashboardHome({ activeDataset, datasets, jobs, loading, onUpload, onOpenDataset, onOpenOverview, onOpenHistory }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const activeId = activeDataset?.dataset_id
  const recentJobs = jobs.slice(0, 5)
  return (
    <div className="home-dashboard">
      <section className="home-welcome">
        <div>
          <span className="home-eyebrow">WORKSPACE</span>
          <h2>Good work starts with good data.</h2>
          <p>Manage datasets, continue analysis, and keep your machine-learning experiments in one place.</p>
        </div>
        <button className="primary home-create-button" type="button" onClick={() => fileInput.current?.click()}>
          <FilePlus2 size={16} /> Create new task
        </button>
        <input ref={fileInput} className="hidden-upload" type="file" accept=".csv" onChange={event => { onUpload(event.target.files?.[0]); event.currentTarget.value = '' }} />
      </section>

      <section className="home-summary-grid" aria-label="Workspace summary">
        <div className="home-summary-card"><span>Datasets</span><strong>{datasets.length}</strong><small>Available in your workspace</small></div>
        <div className="home-summary-card"><span>Experiments</span><strong>{jobs.length}</strong><small>Training runs and comparisons</small></div>
        <div className="home-summary-card"><span>Saved models</span><strong>{jobs.filter(job => job.status === 'completed' && job.model_id).length}</strong><small>Completed model artifacts</small></div>
      </section>

      {activeDataset && (
        <section className="home-active-task">
          <div className="home-section-heading"><div><span className="home-eyebrow">CONTINUE WORKING</span><h3>Current task</h3></div><span className="home-active-status">Active dataset</span></div>
          <div className="home-active-file"><span className="home-file-icon"><Database size={20}/></span><div><strong>{activeDataset.filename}</strong><small>{activeDataset.rows ?? '—'} rows · {activeDataset.columns ?? '—'} columns · {activeDataset.missing_total ?? 0} missing cells</small></div><button className="secondary" type="button" onClick={onOpenOverview}>Open task <ArrowRight size={14}/></button></div>
        </section>
      )}

      <div className="home-content-grid">
        <section className="home-panel">
          <div className="home-section-heading"><div><span className="home-eyebrow">YOUR WORK</span><h3>Recent datasets</h3></div><span className="home-count">{datasets.length} total</span></div>
          {loading ? <p className="home-empty">Loading your datasets…</p> : datasets.length === 0 ? (
            <div className="home-empty-state"><FolderOpen size={27}/><strong>No datasets yet</strong><p>Upload a CSV file to create your first analysis task.</p><button className="secondary" type="button" onClick={() => fileInput.current?.click()}>Choose a dataset</button></div>
          ) : datasets.slice(0, 8).map(dataset => (
            <div className="home-list-row" key={dataset.dataset_id}>
              <span className="home-file-icon"><Database size={17}/></span>
              <div><strong>{dataset.filename}</strong><small>{dataset.saved_at ? new Date(dataset.saved_at).toLocaleString() : 'Saved dataset'}{dataset.dataset_id === activeId ? ' · Current task' : ''}</small></div>
              <button className="secondary" type="button" onClick={() => onOpenDataset(dataset.dataset_id)}>{dataset.dataset_id === activeId ? 'Continue' : 'Open'}</button>
            </div>
          ))}
        </section>

        <section className="home-panel">
          <div className="home-section-heading"><div><span className="home-eyebrow">MODEL WORK</span><h3>Recent experiments</h3></div><button className="home-text-action" type="button" onClick={onOpenHistory}>View history <ArrowRight size={13}/></button></div>
          {recentJobs.length === 0 ? (
            <div className="home-empty-state"><FlaskConical size={27}/><strong>No experiments yet</strong><p>Your model training runs will appear here.</p></div>
          ) : recentJobs.map(job => (
            <div className="home-list-row" key={job.job_id}>
              <span className="home-file-icon"><FlaskConical size={17}/></span>
              <div><strong>{job.best_model || job.model_name || 'Model experiment'}</strong><small>{job.task || 'Task'} · {job.target || 'Target not set'} · {job.created_at ? new Date(job.created_at).toLocaleDateString() : 'Date unavailable'}</small></div>
              <span className={'home-job-status '+String(job.status || 'unknown').toLowerCase()}>{job.status || 'Unknown'}</span>
            </div>
          ))}
        </section>
      </div>
      <p className="home-footnote"><Clock3 size={13}/> Your workspace keeps datasets and experiment history available between sessions.</p>
    </div>
  )
}
