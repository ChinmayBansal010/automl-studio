import { useRef, useState } from 'react'
import {
  Activity, ArrowDownUp, ArrowRight, BarChart3, BrainCircuit, CheckCircle2, Clock3,
  Database, FilePlus2, FlaskConical, FolderOpen, Gauge, RefreshCw, Search, Upload, X
} from 'lucide-react'

type Dataset = { dataset_id: string; filename: string; saved_at?: string }
type Job = {
  job_id: string
  best_model?: string
  model_name?: string
  task?: string
  target?: string
  status?: string
  created_at?: string
  model_id?: string
  model_available?: boolean
}
type ActiveDataset = {
  dataset_id: string
  filename: string
  rows?: number
  columns?: number
  missing_total?: number
} | null

type Props = {
  activeDataset: ActiveDataset
  datasets: Dataset[]
  jobs: Job[]
  loading: boolean
  onUpload: (file?: File) => void
  onOpenDataset: (id: string) => void
  onOpenTab: (tab: string) => void
}

const workflow = [
  { id: 'overview', title: 'Understand data', description: 'Inspect columns, data types and quality.', icon: Gauge, step: '01' },
  { id: 'prepare', title: 'Prepare features', description: 'Handle missing values and transform columns.', icon: RefreshCw, step: '02' },
  { id: 'charts', title: 'Explore visually', description: 'Build charts and inspect distributions.', icon: BarChart3, step: '03' },
  { id: 'train', title: 'Train models', description: 'Configure a task and compare algorithms.', icon: FlaskConical, step: '04' },
  { id: 'evaluation', title: 'Evaluate results', description: 'Review metrics and model performance.', icon: Activity, step: '05' },
  { id: 'predict', title: 'Generate predictions', description: 'Apply a trained model to new data.', icon: BrainCircuit, step: '06' },
]

function dateLabel(value?: string) {
  if (!value) return 'Date unavailable'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

export function DashboardHome({ activeDataset, datasets, jobs, loading, onUpload, onOpenDataset, onOpenTab }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [datasetQuery, setDatasetQuery] = useState('')
  const [datasetSort, setDatasetSort] = useState<'recent' | 'name'>('recent')
  const [experimentFilter, setExperimentFilter] = useState('all')
  const savedModels = jobs.filter(job => Boolean(job.model_id)).length
  const completedRuns = jobs.filter(job => job.status === 'completed').length
  const filteredDatasets = datasets.filter(dataset => dataset.filename.toLowerCase().includes(datasetQuery.trim().toLowerCase())).slice().sort((a, b) => datasetSort === 'name' ? a.filename.localeCompare(b.filename) : String(b.saved_at || '').localeCompare(String(a.saved_at || '')))
  const filteredJobs = jobs.filter(job => experimentFilter === 'all' || String(job.status || 'unknown').toLowerCase() === experimentFilter)
  const recentJobs = filteredJobs.slice(0, 8)

  function acceptFile(file?: File) {
    if (file) onUpload(file)
  }

  return (
    <div className="home-dashboard">
      <section className="home-page-heading">
        <div>
          <div className="home-breadcrumb">WORKSPACE <span>/</span> HOME</div>
          <h2>Your workspace</h2>
          <p>Manage datasets and move through each stage of your machine-learning workflow.</p>
        </div>
        <button className="primary home-upload-button" type="button" onClick={() => fileInput.current?.click()}>
          <FilePlus2 size={16} /> Upload dataset
        </button>
        <input ref={fileInput} className="hidden-upload" type="file" accept=".csv" onChange={event => { acceptFile(event.target.files?.[0]); event.currentTarget.value = '' }} />
      </section>

      <section className="home-stat-grid" aria-label="Workspace statistics">
        <div className="home-stat"><span className="home-stat-icon"><Database size={17}/></span><div><small>Datasets</small><strong>{datasets.length}</strong></div><span className="home-stat-note">In workspace</span></div>
        <div className="home-stat"><span className="home-stat-icon"><FlaskConical size={17}/></span><div><small>Experiments</small><strong>{jobs.length}</strong></div><span className="home-stat-note">Training runs</span></div>
        <div className="home-stat"><span className="home-stat-icon"><CheckCircle2 size={17}/></span><div><small>Completed runs</small><strong>{completedRuns}</strong></div><span className="home-stat-note">Successfully finished</span></div>
        <div className="home-stat"><span className="home-stat-icon"><BrainCircuit size={17}/></span><div><small>Saved models</small><strong>{savedModels}</strong></div><span className="home-stat-note">Available artifacts</span></div>
      </section>

      {activeDataset ? (
        <section className="home-current-task">
          <div className="home-current-label"><span className="home-live-dot"/> CURRENT TASK</div>
          <div className="home-current-main">
            <span className="home-dataset-icon"><Database size={22}/></span>
            <div className="home-current-info">
              <strong title={activeDataset.filename}>{activeDataset.filename}</strong>
              <span>{activeDataset.rows?.toLocaleString() ?? '—'} rows <i/> {activeDataset.columns?.toLocaleString() ?? '—'} columns <i/> {(activeDataset.missing_total ?? 0).toLocaleString()} missing cells</span>
            </div>
            <button className="secondary home-continue" type="button" onClick={() => onOpenTab('overview')}>Continue analysis <ArrowRight size={15}/></button>
          </div>
        </section>
      ) : (
        <section className={'home-upload-panel' + (dragging ? ' is-dragging' : '')}
          onDragOver={event => { event.preventDefault(); setDragging(true) }}
          onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false) }}
          onDrop={event => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files?.[0]) }}>
          <span className="home-upload-icon"><Upload size={20}/></span>
          <div><strong>Start with a dataset</strong><p>Drop a CSV file here, or choose one from your device. Your analysis starts after the upload completes.</p></div>
          <button className="secondary" type="button" onClick={() => fileInput.current?.click()}>Choose CSV file</button>
        </section>
      )}

      <section className="home-workflow-section">
        <div className="home-section-title">
          <div><span>WORKFLOW</span><h3>From dataset to predictions</h3></div>
          <p>{activeDataset ? 'Choose a stage to continue.' : 'Upload a dataset to unlock the analysis steps.'}</p>
        </div>
        <div className="home-workflow-grid">
          {workflow.map(({ id, title, description, icon: Icon, step }) => (
            <button className="home-workflow-card" type="button" key={id} disabled={!activeDataset}
              onClick={() => onOpenTab(id)}>
              <span className="home-workflow-top"><span className="home-workflow-icon"><Icon size={18}/></span><small>{step}</small></span>
              <strong>{title}</strong><span>{description}</span>
              <span className="home-workflow-link">Open step <ArrowRight size={13}/></span>
            </button>
          ))}
        </div>
      </section>

      <div className="home-lower-grid">
        <section className="home-data-panel">
          <div className="home-panel-heading"><div><span>DATA LIBRARY</span><h3>Recent datasets</h3></div><span className="home-total">{datasets.length} total</span></div>
          <div className="home-library-controls">
            <label className="home-search"><Search size={15}/><input aria-label="Search datasets" value={datasetQuery} onChange={event => setDatasetQuery(event.target.value)} placeholder="Search datasets by filename"/></label>
            <button className="home-sort-button" type="button" onClick={() => setDatasetSort(current => current === 'recent' ? 'name' : 'recent')}><ArrowDownUp size={14}/>{datasetSort === 'recent' ? 'Most recent' : 'Name A–Z'}</button>
          </div>
          {loading ? <div className="home-loading">Loading workspace data…</div> : datasets.length === 0 ? (
            <div className="home-empty"><FolderOpen size={22}/><strong>No datasets saved</strong><p>Upload a CSV to start exploring, preparing and training models.</p><button className="secondary" type="button" onClick={() => fileInput.current?.click()}>Upload a dataset</button></div>
          ) : filteredDatasets.length === 0 ? (
            <div className="home-empty"><Search size={22}/><strong>No matching datasets</strong><p>Try another filename or clear your search.</p><button className="home-view-all" type="button" onClick={() => setDatasetQuery('')}><X size={13}/> Clear search</button></div>
          ) : filteredDatasets.map(dataset => (
            <div className="home-dataset-row" key={dataset.dataset_id}>
              <span className="home-row-icon"><Database size={16}/></span>
              <div><strong title={dataset.filename}>{dataset.filename}</strong><small>{dateLabel(dataset.saved_at)}{dataset.dataset_id === activeDataset?.dataset_id ? ' · Active dataset' : ''}</small></div>
              <button className="home-row-action" type="button" onClick={() => onOpenDataset(dataset.dataset_id)}>{dataset.dataset_id === activeDataset?.dataset_id ? 'Continue' : 'Open'} <ArrowRight size={13}/></button>
            </div>
          ))}
        </section>

        <section className="home-data-panel">
          <div className="home-panel-heading"><div><span>MODEL ACTIVITY</span><h3>Recent experiments</h3></div><button className="home-view-all" type="button" onClick={() => onOpenTab('experiments')}>View history <ArrowRight size={13}/></button></div>
          <div className="home-filter-row" role="group" aria-label="Filter experiments">
            {['all', 'completed', 'running', 'failed', 'queued'].map(status => <button key={status} type="button" className={experimentFilter === status ? 'active' : ''} onClick={() => setExperimentFilter(status)}>{status === 'all' ? 'All' : status[0].toUpperCase() + status.slice(1)}</button>)}
          </div>
          {recentJobs.length === 0 ? (
            <div className="home-empty"><FlaskConical size={22}/><strong>{jobs.length ? 'No experiments in this status' : 'No experiments yet'}</strong><p>{jobs.length ? 'Choose another filter to see more runs.' : 'Configure a training run to compare algorithms and track results.'}</p>{jobs.length > 0 && <button className="home-view-all" type="button" onClick={() => setExperimentFilter('all')}>Show all experiments</button>}</div>
          ) : recentJobs.map(job => (
            <div className="home-experiment-row" key={job.job_id}>
              <span className="home-row-icon"><FlaskConical size={16}/></span>
              <div><strong>{job.best_model || job.model_name || 'Model experiment'}</strong><small>{job.task || 'Task'} · {job.target || 'Target not set'} · {dateLabel(job.created_at)}</small></div>
              <span className={'home-status status-' + String(job.status || 'unknown').toLowerCase().replace(/[^a-z]+/g, '-')}>{job.status || 'Unknown'}</span>
            </div>
          ))}
        </section>
      </div>
      <p className="home-footnote"><Clock3 size={13}/> Showing {filteredDatasets.length} of {datasets.length} datasets · {recentJobs.length} of {filteredJobs.length} matching experiments.</p>
    </div>
  )
}
