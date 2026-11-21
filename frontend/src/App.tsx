import { useEffect, useRef, useState } from 'react'
import { Activity, BarChart3, BrainCircuit, ChevronLeft, ChevronRight, Database, Download, FilePlus2, FlaskConical, Gauge, LogOut, RefreshCw, Sparkles, Target, UploadCloud } from 'lucide-react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, Pie, PieChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { AuthScreen } from './AuthScreen'
import { authClient } from './auth'

const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')).replace(/\/+$/, '')
const API = `${BACKEND_URL}/api`
let nextApiRequestAt = 0
async function paceApiRequest(): Promise<void> {
  const now = Date.now()
  const wait = Math.max(0, nextApiRequestAt - now)
  nextApiRequestAt = Math.max(now, nextApiRequestAt) + 100
  if (wait > 0) await new Promise(resolve => setTimeout(resolve, wait))
}

async function getValidToken(forceRefresh = false): Promise<string> {
  const readSession = (force: boolean) => authClient.getSession(
    force ? { fetchOptions: { headers: { 'X-Force-Fetch': 'true' } } } : undefined
  )

  let result = await readSession(forceRefresh)
  if (!result.error && result.data?.session?.token) return result.data.session.token

  if (!forceRefresh) {
    result = await readSession(true)
    if (!result.error && result.data?.session?.token) return result.data.session.token
  }

  throw new Error('Your session expired. Please sign in again.')
}

async function fetchWithRetry(url: string, init: RequestInit, retries = 2): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(url, init)
      // Retry on 502/503/504 (gateway / unavailable) but not on last attempt
      if (attempt < retries && r.status >= 502 && r.status <= 504) {
        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)))
        continue
      }
      return r
    } catch (err) {
      if (attempt >= retries) throw err
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)))
    }
  }
  return fetch(url, init) // unreachable but satisfies TS
}

async function api(url: string, init?: RequestInit): Promise<any> {
  await paceApiRequest()
  if (!BACKEND_URL) throw new Error('Backend URL is not configured. Set VITE_BACKEND_URL for this deployment.')
  const token = await getValidToken()
  const headers = new Headers(init?.headers)
  headers.set('Authorization', 'Bearer ' + token)
  let r = await fetchWithRetry(API + url, { ...init, headers })
  // If we get a 401, try refreshing the token once and retry the request
  if (r.status === 401) {
    let freshToken: string
    try { freshToken = await getValidToken(true) } catch { freshToken = '' }
    if (freshToken && freshToken !== token) {
      headers.set('Authorization', 'Bearer ' + freshToken)
      r = await fetchWithRetry(API + url, { ...init, headers })
    }
  }
  const b = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(b.detail || ('Request failed ' + r.status))
  return b
}
type DatasetBackup={key:string;userId:string;datasetId:string;filename:string;blob:Blob;updatedAt:string}
const BACKUP_DB='automl-studio-browser-backups'
function openBackupDb():Promise<IDBDatabase>{
 return new Promise((resolve,reject)=>{
  const request=indexedDB.open(BACKUP_DB,1)
  request.onupgradeneeded=()=>{const db=request.result;if(!db.objectStoreNames.contains('datasets'))db.createObjectStore('datasets',{keyPath:'key'})}
  request.onsuccess=()=>resolve(request.result)
  request.onerror=()=>reject(request.error)
 })
}
async function saveDatasetBackup(userId:string,datasetId:string,filename:string,blob:Blob):Promise<void>{
 const db=await openBackupDb()
 await new Promise<void>((resolve,reject)=>{const tx=db.transaction('datasets','readwrite');tx.objectStore('datasets').put({key:userId+':'+datasetId,userId,datasetId,filename,blob,updatedAt:new Date().toISOString()} satisfies DatasetBackup);tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)})
 db.close()
}
async function readDatasetBackup(userId:string,datasetId:string):Promise<DatasetBackup|null>{
 const db=await openBackupDb()
 const result=await new Promise<DatasetBackup|null>((resolve,reject)=>{const request=db.transaction('datasets','readonly').objectStore('datasets').get(userId+':'+datasetId);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error)})
 db.close()
 return result
}
function setActiveDatasetRef(userId:string,datasetId:string,filename:string):void{try{localStorage.setItem('automl-active-dataset:'+userId,JSON.stringify({datasetId,filename}))}catch{}}
function getActiveDatasetRef(userId:string):{datasetId:string;filename:string}|null{try{const value=localStorage.getItem('automl-active-dataset:'+userId);return value?JSON.parse(value):null}catch{return null}}
const fmt=(v:any)=>v===null||v===undefined?'—':typeof v==='number'?Number(v.toFixed?.(4)??v).toLocaleString():String(v)
function Workspace({user,onSignOut}:{user:{id:string;email:string};onSignOut:()=>Promise<void>}){
 const uploadInput=useRef<HTMLInputElement>(null)
 const evaluationRef=useRef<HTMLDivElement>(null)
 const [initialLoading,setInitialLoading]=useState(true)
 const [tab,S]=useState('overview'),[p,SP]=useState<any>(null),[eda,SE]=useState<any>(null),[err,ER]=useState(''),[busy,SB]=useState(false),[ws,SW]=useState<any>({saved_datasets:[],saved_models:0}),[jobs,SJ]=useState<any[]>([])
 const [op,SO]=useState('fill_missing'),[strategy,SS]=useState('median'),[cols,SC]=useState<string[]>([]),[threshold,ST]=useState(.5),[newName,SN]=useState(''),[value,SV]=useState(''),[components,SCM]=useState(2),[sampleSize,SSample]=useState(1000)
 const [kind,SK]=useState('bar'),[x,SX]=useState(''),[y,SY]=useState(''),[agg,SA]=useState('mean'),[rows,SR]=useState(20),[grid,SG]=useState(1),[chart,SCH]=useState<any>(null)
 const [target,STG]=useState(''),[task,STK]=useState('classification'),[mode,SM]=useState('auto'),[modelName,SNModel]=useState('Random Forest'),[selectedModels,SSelectedModels]=useState<string[]>([]),[split,SSplit]=useState(.2),[selection,SSEL]=useState('none'),[k,SKN]=useState(10),[pca,SPCA]=useState(false),[pcaK,SPK]=useState(3),[scale,SSC]=useState(true),[drop,SD]=useState(false),[features,SF]=useState<string[]>([]),[model,SModel]=useState<any>(null),[prediction,SPred]=useState<any>(null),[trainingQueue,STQ]=useState<any[]>([])
 const models=task==='classification'?['Logistic Regression','Decision Tree','Random Forest','Extra Trees','Gradient Boosting','Histogram Gradient Boosting','AdaBoost','K-Nearest Neighbors','Support Vector Machine','Naive Bayes','Neural Network (MLP)']:['Ridge Regression','Decision Tree','Random Forest','Extra Trees','Gradient Boosting','Histogram Gradient Boosting','AdaBoost','K-Nearest Neighbors','Support Vector Regression','Neural Network (MLP)']
 const targetColumns=(p?.columns_info||[]).filter((c:any)=>c.unique>=2&&(task==='regression'?c.kind==='numeric':c.unique<=100))
 const chartXColumns=(p?.columns_info||[]).filter((c:any)=>kind==='heatmap'?false:['histogram','scatter','density'].includes(kind)?c.kind==='numeric':true)
 const chartYColumns=(p?.columns_info||[]).filter((c:any)=>['line','area','scatter','bar','box','radar'].includes(kind)?c.kind==='numeric':false)
 const activeTrainingJob=trainingQueue.find((j:any)=>j.status==='queued'||j.status==='running')
 const hasActiveTraining=Boolean(activeTrainingJob)
 const prepColumns=(p?.columns_info||[]).filter((c:any)=>((op==='drop_outliers'||op==='clip_outliers'||(op==='fill_missing'&&['mean','median','zero'].includes(strategy)))?c.kind==='numeric':true))
 const selectedNumericCount=cols.filter((c:string)=>(p?.columns_info||[]).some((info:any)=>info.column===c&&info.kind==='numeric')).length
 const needsSelection=['drop_columns','convert_numeric','convert_datetime','rename_column'].includes(op)
 const preprocessCanRun=(!needsSelection||(op==='rename_column'?cols.length===1:cols.length>0))&&!((op==='drop_outliers'||op==='clip_outliers'||(op==='fill_missing'&&['mean','median','zero'].includes(strategy)))&&prepColumns.length===0)
 const needsColumns=['fill_missing','drop_missing_rows','drop_columns','drop_outliers','clip_outliers','convert_numeric','convert_datetime','rename_column'].includes(op)
 const bestResult=model?.results?.find((r:any)=>r.model_name===model.best_model)
 const actualPredictedRows=bestResult?(bestResult.actual||[]).map((actual:any,i:number)=>({Actual:actual,Predicted:bestResult.predictions?.[i]??null,...(model.task==='classification'?{Correct:actual===bestResult.predictions?.[i]?'Yes':'No'}:{})})):[]
 const regressionPoints=bestResult?(bestResult.actual||[]).map((actual:any,i:number)=>({actual,predicted:bestResult.predictions?.[i]})).filter((v:any)=>v.actual!==null&&v.actual!==undefined&&v.predicted!==null&&v.predicted!==undefined&&Number.isFinite(Number(v.actual))&&Number.isFinite(Number(v.predicted))).map((v:any)=>({actual:Number(v.actual),predicted:Number(v.predicted)})):[]
 const confusionRows=(cm:any)=>cm?[
   {'Actual class ↓ / Predicted class →':'',...Object.fromEntries(cm.labels.map((label:string)=>[label,label]))},
   ...cm.labels.map((label:string,i:number)=>({'Actual class ↓ / Predicted class →':label,...Object.fromEntries(cm.labels.map((predicted:string,j:number)=>[predicted,cm.matrix[i]?.[j]??0]))}))
 ]:[]
 useEffect(()=>{SSelectedModels(models)},[task])
 useEffect(()=>{if(!p)return;const xs=chartXColumns;if(xs.length&&!xs.some((c:any)=>c.column===x))SX(xs[0].column);if(!xs.length)SX('');const ys=chartYColumns;if(ys.length&&!ys.some((c:any)=>c.column===y))SY(ys[0].column);if(!ys.length)SY('')},[kind,p])

 async function refresh(){const a=await api('/workspace'),b=await api('/workspace/jobs');SW(a);SJ(b.jobs||[])}
 useEffect(()=>{(async()=>{await refresh();await recoverActiveDataset()})().catch(error=>ER(error instanceof Error?error.message:String(error))).finally(()=>setInitialLoading(false))},[])
 async function signOut(){try{await onSignOut()}catch(error){ER(String(error))}}
 async function cacheDatasetBackup(datasetId:string,filename:string){
  try{const token=await getValidToken();const response=await fetchWithRetry(API+'/datasets/'+datasetId+'/download',{headers:{Authorization:'Bearer '+token}});if(response.ok)await saveDatasetBackup(user.id,datasetId,filename,await response.blob())}catch{}
 }
 async function loaded(d:any){ER('');SP(d);setActiveDatasetRef(user.id,d.dataset_id,d.filename);SC(d.columns_info.map((c:any)=>c.column));SF(d.columns_info.map((c:any)=>c.column));SX(d.columns_info[0]?.column||'');SY(d.types?.numeric?.[0]||d.columns_info[1]?.column||'');STG('');SModel(null);SPred(null);SCH([]);await cacheDatasetBackup(d.dataset_id,d.filename);SE(await api('/datasets/'+d.dataset_id+'/eda'));await refresh()}
 async function recoverActiveDataset(){
  const active=getActiveDatasetRef(user.id);if(!active)return
  try{const profile=await api('/datasets/'+active.datasetId+'/profile');await loaded({...profile,filename:active.filename})}
  catch(error){
   const backup=await readDatasetBackup(user.id,active.datasetId)
   if(!backup)throw error
   const form=new FormData();form.append('file',backup.blob,backup.filename)
   const restored=await api('/datasets/'+active.datasetId+'/restore',{method:'POST',body:form})
   await loaded({...restored,filename:backup.filename})
  }
 }
 async function ensureDatasetReady(){
  if(!p)return
  try{await api('/datasets/'+p.dataset_id+'/profile')}
  catch(error){
   const backup=await readDatasetBackup(user.id,p.dataset_id)
   if(!backup)throw error
   const form=new FormData();form.append('file',backup.blob,backup.filename)
   const restored=await api('/datasets/'+p.dataset_id+'/restore',{method:'POST',body:form})
   SP({...restored,filename:p.filename});setActiveDatasetRef(user.id,p.dataset_id,p.filename);await cacheDatasetBackup(p.dataset_id,p.filename);SE(await api('/datasets/'+p.dataset_id+'/eda'))
  }
 }
 async function upload(file?:File){if(!file)return;SB(true);ER('');try{const f=new FormData();f.append('file',file);const dataset=await api('/datasets',{method:'POST',body:f});try{await saveDatasetBackup(user.id,dataset.dataset_id,file.name,file)}catch{}setActiveDatasetRef(user.id,dataset.dataset_id,file.name);await loaded(dataset);S('overview')}catch(e){ER(String(e))}finally{SB(false)}}
 async function restore(id:string){SB(true);try{await loaded(await api('/workspace/datasets/'+id+'/load',{method:'POST'}));S('overview')}catch(e){ER(String(e))}finally{SB(false)}}
 async function prep(){if(!p)return;SB(true);ER('');try{await ensureDatasetReady();const r=await api('/datasets/'+p.dataset_id+'/preprocess',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation:op,columns:cols,strategy,value,threshold,new_name:newName,sample_size:sampleSize})});await loaded({...r,filename:p.filename})}catch(e){ER(String(e))}finally{SB(false)}}
 async function feat(operation:string){if(!p)return;SB(true);ER('');try{await ensureDatasetReady();const r=await api('/datasets/'+p.dataset_id+'/feature-engineering',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,columns:cols,components})});await loaded({...r,filename:p.filename})}catch(e){ER(String(e))}finally{SB(false)}}
 async function makeChart(){if(!p)return;if(kind!=='heatmap'&&!x){ER('Select a valid X-axis column.');return}if(['line','area','scatter','box','radar'].includes(kind)&&!y){ER('Select a valid numeric Y-axis column.');return}SB(true);ER('');try{await ensureDatasetReady();const generated=await api('/datasets/'+p.dataset_id+'/chart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({chart_type:kind,x:kind==='heatmap'?null:x,y:['pie','count','histogram','heatmap'].includes(kind)?null:y,aggregation:agg,rows,columns:grid})});SCH((current:any)=>[...(Array.isArray(current)?current:current?[current]:[]),generated])}catch(e){ER(String(e))}finally{SB(false)}}
 async function watchTraining(jobId:string){try{while(true){await new Promise(resolve=>setTimeout(resolve,2000));const status=await api('/training-jobs/'+jobId);STQ(q=>q.map(j=>j.job_id===jobId?{...j,...status}:j));if(status.status==='completed'){if(status.result){SModel(status.result);S('evaluation');window.setTimeout(()=>evaluationRef.current?.scrollIntoView({behavior:'smooth',block:'start'}),80);await refresh()}break}if(status.status==='failed'){ER(status.error||'Training failed.');break}if(status.status==='cancelled'){STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:'cancelled'}:j));break}}}catch(e){const detail=e instanceof Error?e.message:String(e);STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:'failed',error:detail}:j));try{await recoverActiveDataset()}catch{}ER('The training service became unavailable before it could return a result. Render may have restarted after running out of memory or waking from sleep. Your active CSV is backed up in this browser. Retry with fewer selected models, fewer input features, or a smaller dataset.')} }
 async function train(){if(!p||!target){ER('Select a valid target column.');return}if(!features.some(c=>c!==target)){ER('Select at least one input feature.');return}SB(true);ER('');try{await ensureDatasetReady();const queued=await api('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:p.dataset_id,target,task,mode,model_name:modelName,model_names:mode==='manual'?selectedModels:undefined,test_size:split,scale_numeric:scale,drop_high_missing:drop,selected_features:features.filter(c=>c!==target),feature_selection:selection,k_features:k,use_pca:pca,pca_components:pcaK})});STQ(q=>[{job_id:queued.job_id,status:queued.status,queue_position:queued.queue_position,target,task,model_name:mode==='auto'?'All '+task+' models':selectedModels.join(', '),created_at:new Date().toISOString()},...q].slice(0,10));void watchTraining(queued.job_id)}catch(e){ER(String(e))}finally{SB(false)}}
 async function cancelTraining(jobId:string){try{const result=await api('/training-jobs/'+jobId+'/cancel',{method:'POST'});STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:result.status==='cancelled'?'cancelled':'running',cancel_requested:result.status==='cancelling',message:result.message}:j))}catch(e){ER(e instanceof Error?e.message:String(e))}}
 async function predict(file?:File){if(!file||!model)return;SB(true);try{const f=new FormData();f.append('file',file);f.append('model_id',model.model_id);SPred(await api('/predict',{method:'POST',body:f}))}catch(e){ER(String(e))}finally{SB(false)}}
 async function download(url:string,filename?:string){try{if(!BACKEND_URL)throw new Error('Backend URL is not configured. Set VITE_BACKEND_URL for this deployment.');const token=await getValidToken();const response=await fetchWithRetry(url.startsWith('/')?BACKEND_URL+url:API+url,{headers:{Authorization:'Bearer '+token}});if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(body.detail||('Download failed '+response.status))}const disposition=response.headers.get('content-disposition')||'';const match=disposition.match(/filename\*?=(?:UTF-8''|")?([^";]+)/i);const serverFilename=match?.[1]?decodeURIComponent(match[1].replace(/"/g,'').trim()):undefined;const objectUrl=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=objectUrl;a.download=filename||serverFilename||'download';document.body.appendChild(a);a.click();a.remove();window.setTimeout(()=>URL.revokeObjectURL(objectUrl),1000)}catch(e){ER(e instanceof Error?e.message:String(e))}}
 const nav=[['overview','Overview',Gauge],['prepare','Preprocess data',RefreshCw],['charts','Chart builder',BarChart3],['train','Model training',FlaskConical],['evaluation','Evaluation',Activity],['predict','Predictions',Target],['models','Model library',BrainCircuit]] as any[]
 const phaseIndex=Math.max(0,nav.findIndex(t=>t[0]===tab))
 const goToPhase=(index:number)=>S(nav[index]?.[0]||'overview')
 const buttonFeedback=(event:any)=>{const button=(event.target as HTMLElement).closest('button') as HTMLButtonElement|null;if(!button||button.disabled)return;button.dataset.working='true';window.setTimeout(()=>{delete button.dataset.working},900)}
 const card=(title:string,body:any)=><section className="card"><h2>{title}</h2>{body}</section>
 const field=(label:string,child:any)=><label className="field"><span>{label}</span>{child}</label>
 const metric=(label:string,val:any)=><div className="metric"><small>{label}</small><strong>{fmt(val)}</strong></div>
 const tableContent=(data:any[])=> <div className="tablewrap"><table><thead><tr>{Object.keys(data[0]).map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>{data.slice(0,100).map((r:any,i:number)=><tr key={i}>{Object.keys(data[0]).map(c=><td key={c}>{r[c]===null||r[c]===undefined?'—':String(r[c])}</td>)}</tr>)}</tbody></table></div>
 const table=(data:any[])=>!data?.length?<p>No rows to display.</p>:data.length>12?<details className="table-disclosure"><summary>Show table · {data.length.toLocaleString()} rows (first 100 displayed)</summary>{tableContent(data)}</details>:tableContent(data)
 return <div className="app" onClickCapture={buttonFeedback}><aside className="sidebar"><div className="brand"><Sparkles/><div><b>AutoML Studio</b><small>PRIVATE DATA WORKSPACE</small></div></div><nav>{nav.map(([id,label,Icon])=><button key={id} className={tab===id?'active':''} onClick={()=>S(id)}><Icon size={17}/>{label}</button>)}</nav><div className="sidebottom"><span className="signed-in">{user.email}</span><button className="signout" onClick={signOut}><LogOut size={14}/>Sign out</button></div></aside><main><header className="workspace-header"><div className="header-title"><small>MACHINE LEARNING WORKSPACE</small><h1>{nav.find(t=>t[0]===tab)?.[1]}</h1>{p&&<span className="filetag header-filetag">{p.filename}</span>}</div><div className="header-actions"><button className="new-task-button" onClick={()=>uploadInput.current?.click()} disabled={busy||hasActiveTraining}><FilePlus2 size={16}/>New task</button><input ref={uploadInput} className="hidden-upload" type="file" accept=".csv" onChange={e=>{const file=e.target.files?.[0];if(file)void upload(file);e.currentTarget.value=''}} /></div></header><section className="content">
 {err&&<div className="error">{err}<button onClick={()=>ER('')}>Dismiss</button></div>}{busy&&<div className="busy-panel"><span className="loader-ring"/><div><strong>Working on your request</strong><small>Processing data and updating the workspace…</small></div><span className="busy-pulse"/></div>}{initialLoading&&!p&&<div className="skeleton-dashboard" aria-label="Loading workspace"><div className="skeleton-metrics">{[1,2,3,4].map(i=><div className="skeleton skeleton-metric" key={i}/>)}</div><div className="skeleton skeleton-card"/><div className="skeleton skeleton-card short"/></div>}{p&&tab!=='overview'&&<div className="dataset-context"><div><small>ACTIVE DATASET · LIVE PROFILE</small><strong>{p.filename}</strong><span>{p.rows??p.shape?.[0]??'—'} rows · {p.columns??p.shape?.[1]??p.columns_info?.length??'—'} columns · {p.missing_total??0} missing cells</span></div><button className="secondary" onClick={()=>S('overview')}>View overview</button></div>}
 {!p&&!initialLoading&&<div className="welcome"><div><small>AUTOMATED MACHINE LEARNING</small><h2>Turn raw data into<br/><em>predictive insights.</em></h2><p>Upload CSV data to profile, preprocess, visualize, train models, and predict locally.</p><label className="upload"><UploadCloud/> Choose CSV<input type="file" accept=".csv" onChange={e=>upload(e.target.files?.[0])}/></label></div><div className="hero-card"><Database size={40}/><h3>CSV → Insights</h3><p>Data quality, preprocessing, charts, AutoML, evaluation and predictions.</p></div></div>}
 {p&&tab==='overview'&&<div className="stack"><div className="metrics">{metric('Shape',(p.shape||[p.rows,p.columns]).join(' × '))}{metric('Missing cells',p.missing_total)}{metric('Duplicate rows',p.duplicate_rows)}{metric('Memory MB',p.memory_mb)}</div>{card('Dataset schema',table(p.columns_info.map((c:any)=>({column:c.column,type:c.dtype,missing:c.missing,missing_pct:c.missing_pct+'%',unique:c.unique,sample:(c.sample||[]).join(', ')}))))}{card('Detailed numeric statistics',table(Object.entries(p.column_stats||{}).map(([column,s]:any)=>({column,...s}))))}{card('Data preview',table(p.preview))}{card('Saved local workspace',<><div className="metrics">{metric('Datasets',ws.saved_datasets?.length||0)}{metric('Models',ws.saved_models||0)}{metric('Training jobs',jobs.length)}</div>{(ws.saved_datasets||[]).map((d:any)=><div className="savedrow" key={d.dataset_id}><span>{d.filename}<small>{d.saved_at}</small></span><button className="secondary" onClick={()=>restore(d.dataset_id)}>Restore</button></div>)}{jobs.filter((j:any)=>j.model_id&&j.model_available).slice(0,8).map((j:any)=><div className="savedrow" key={j.model_id}><span>{j.best_model} · {j.task} · {j.target}<small>{j.created_at} · {(j.model_size_bytes/1048576).toFixed(2)} MB</small></span><button className="secondary" onClick={()=>download('/api/models/'+j.model_id+'/download')}><Download size={14}/> Download</button></div>)}</>)}</div>}
 {tab==='models'&&<div className="stack">{card('Saved model library',<><p>Download trained model artifacts and reuse them for predictions. Each experiment stores its best-performing model as a portable Joblib pipeline, including its fitted preprocessing steps.</p><div className="metrics">{metric('Saved models',jobs.filter((j:any)=>j.model_id&&j.model_available).length)}{metric('Unavailable artifacts',jobs.filter((j:any)=>j.model_id&&!j.model_available).length)}</div>{jobs.filter((j:any)=>j.model_id).length===0?<p>No saved models yet. Train a model to create your first downloadable artifact.</p>:jobs.filter((j:any)=>j.model_id).map((j:any)=><div className="model-library-item" key={j.model_id}><div className="model-library-info"><strong>{j.best_model||'Trained model'}</strong><span>{j.task||'Task'} · Target: {j.target||'—'}</span><small>{j.created_at||'Date unavailable'} · {j.model_available?(j.model_size_bytes/1048576).toFixed(2)+' MB':'Artifact unavailable on server'}</small></div><button className="secondary" disabled={!j.model_available} onClick={()=>download('/api/models/'+j.model_id+'/download')}><Download size={15}/> Download .joblib</button></div>)}</>)}{card('Using a downloaded model',<><p>The downloaded .joblib file contains the fitted scikit-learn pipeline. Load it in a compatible Python environment with Joblib and the same scikit-learn version used for training.</p><pre className="code-sample">{'import joblib\n\nmodel = joblib.load("your-model.joblib")\npredictions = model["pipeline"].predict(new_data[model["feature_cols"]])'}</pre><p className="hint">Only load model files from sources you trust. Joblib uses Python pickle serialization and can execute code when loading an untrusted file.</p></>)}</div> {p&&tab==='prepare'&&<div className="stack">{card('Step 1 · Preprocessing operation',<><p>{op==='fill_missing'?(strategy==='drop'?'Drop rows containing missing values in selected columns, or all columns when none are selected.':(['mean','median','zero'].includes(strategy)?`Fill selected numeric columns using ${strategy}. If none are selected, all numeric columns are used.`:`Fill missing values using ${strategy} in selected columns, or all columns when none are selected.`)):op==='drop_missing_rows'?'Drop rows with missing values in selected columns, or any column when none are selected.':op==='drop_columns'?'Permanently remove selected columns. Select at least one.':op==='drop_high_missing'?'Remove columns whose missing-value percentage exceeds the selected threshold.':op==='drop_duplicates'?'Remove duplicate rows while keeping the first occurrence.':op==='drop_outliers'?'Remove rows outside the 1.5×IQR range for selected numeric columns, or all numeric columns if none are selected.':op==='clip_outliers'?'Clip selected numeric columns to their 1.5×IQR bounds.':op==='convert_numeric'?'Convert selected columns to numeric values; invalid values become missing.':op==='convert_datetime'?'Convert selected columns to datetime values.':op==='rename_column'?'Select exactly one column and enter a unique new name.':op==='sample_rows'?'Keep a reproducible random sample of the requested number of rows.':'Reset the row index.'} The dataset profile and missing-value counts refresh after a successful operation.</p><div className="formgrid">{field('Operation',<select value={op} onChange={e=>{SO(e.target.value);ER('');SC([])}}>{[['fill_missing','Handle null values'],['drop_duplicates','Remove duplicates'],['drop_missing_rows','Drop rows with nulls'],['drop_columns','Drop selected columns'],['drop_high_missing','Drop high-null columns'],['drop_outliers','Remove IQR outliers'],['clip_outliers','Clip IQR outliers'],['convert_numeric','Convert numeric'],['convert_datetime','Convert datetime'],['rename_column','Rename column'],['sample_rows','Sample rows'],['reset_index','Reset index']].map(([v,l])=><option key={v} value={v}>{l}</option>)}</select>)}{op==='fill_missing'&&field('Null strategy',<select value={strategy} onChange={e=>{SS(e.target.value);ER('');SC([])}}>{['median','mean','mode','constant','zero','ffill','bfill','drop'].map(v=><option key={v}>{v}</option>)}</select>)}{op==='fill_missing'&&strategy==='constant'&&field('Replacement value',<input value={value} onChange={e=>SV(e.target.value)}/>)}{op==='drop_high_missing'&&field('Missing threshold '+Math.round(threshold*100)+'%',<input type="range" min=".1" max="1" step=".05" value={threshold} onChange={e=>ST(+e.target.value)}/>)}{op==='sample_rows'&&field('Rows to keep',<input type="number" min="1" max={p.rows} value={sampleSize} onChange={e=>SSample(Math.max(1,Math.min(p.rows,Number(e.target.value)||1)))}/>)}{op==='rename_column'&&field('New column name',<input value={newName} onChange={e=>SN(e.target.value)}/>)}</div>{needsColumns&&<>{((op==='drop_outliers'||op==='clip_outliers'||(op==='fill_missing'&&['mean','median','zero'].includes(strategy)))&&prepColumns.length===0)&&<p className="hint">This operation requires numeric columns, but the dataset has none available.</p>}<h3>Columns {op==='rename_column'?'(select exactly one)':''}</h3><div className="checkgrid">{prepColumns.map((c:any)=><label key={c.column}><input type="checkbox" checked={cols.includes(c.column)} onChange={e=>SC(a=>e.target.checked?(op==='rename_column'?[c.column]:[...a,c.column]):a.filter(v=>v!==c.column))}/>{c.column} <small>({c.missing} null)</small></label>)}</div></>}<button className="primary" onClick={prep} disabled={busy||hasActiveTraining||!preprocessCanRun}>Apply and save</button></>)}{card('Step 2 · Feature engineering',<><p>Choose columns above, then apply an encoding or numeric transform.</p><div className="actions"><button onClick={()=>feat('one_hot_encode')} disabled={busy||hasActiveTraining||cols.length===0}>One-hot encode</button><button onClick={()=>feat('standardize')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Standardize</button><button onClick={()=>feat('minmax')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Min-max scale</button><button onClick={()=>feat('log_transform')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Log transform</button><button onClick={()=>feat('polynomial')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Squared features</button>{field('PCA components',<input type="number" min="1" max="20" value={components} onChange={e=>SCM(+e.target.value)}/>)}<button onClick={()=>feat('pca')} disabled={busy||hasActiveTraining||selectedNumericCount<2}>Run PCA</button></div></>)}</div>}
 {p&&tab==='charts'&&<div className="stack">
{card('Chart dashboard',<>
<p>Each generated graph is added to this dashboard. Choose a layout to arrange multiple graphs together.</p>
<div className="formgrid">
{field('Graph type',<select value={kind} onChange={e=>SK(e.target.value)}>{['bar','line','area','scatter','pie','count','histogram','heatmap','density','box','radar'].map(v=><option key={v}>{v==='box'?'box plot':v}</option>)}</select>)}
{kind!=='heatmap'&&field(kind==='histogram'?'Numeric column':kind==='scatter'?'Numeric X axis':'X axis / category',<select value={x} onChange={e=>SX(e.target.value)}>{chartXColumns.map((col:any)=><option key={col.column} value={col.column}>{col.column}</option>)}</select>)}
{['line','area','scatter','bar','box','radar'].includes(kind)&&field('Numeric Y axis / value',<select value={y} onChange={e=>SY(e.target.value)}>{chartYColumns.map((col:any)=><option key={col.column} value={col.column}>{col.column}</option>)}</select>)}
{['bar','line','area','radar'].includes(kind)&&field('Aggregation',<select value={agg} onChange={e=>SA(e.target.value)}>{['mean','sum','median','count','min','max'].map(v=><option key={v}>{v}</option>)}</select>)}
{field('Max groups '+rows,<input type="range" min="5" max="100" step="5" value={rows} onChange={e=>SR(+e.target.value)}/>)}
{field('Dashboard columns',<select value={grid} onChange={e=>SG(+e.target.value)}><option value={1}>1 column</option><option value={2}>2 columns</option><option value={3}>3 columns</option></select>)}
</div>
<div className="actions"><button className="primary" onClick={makeChart} disabled={busy||(kind!=='heatmap'&&!x)||(['line','area','scatter'].includes(kind)&&!y)}>Add graph to dashboard</button>{Array.isArray(chart)&&chart.length>0&&<button className="secondary" onClick={()=>SCH([])}>Clear dashboard</button>}</div>
</>)}
{Array.isArray(chart)&&chart.length>0&&<div className="chart-dashboard" style={{gridTemplateColumns:'repeat('+grid+',minmax(0,1fr))'}}>
{chart.map((item:any,index:number)=><section className="card chart-card" key={index}>
<div className="chart-card-heading"><h2>{item.chart_type} chart · {index+1}</h2><button className="secondary" onClick={()=>SCH((current:any[])=>current.filter((_:any,i:number)=>i!==index))}>Remove</button></div>
{item.chart_type==='heatmap'?(item.data?.length?table(item.data):<p>{item.message||'No numeric columns are available for this heatmap.'}</p>):<ResponsiveContainer width="100%" height={320}>
{item.chart_type==='pie'?<PieChart><Tooltip/><Legend/><Pie data={item.data} dataKey="value" nameKey="x" outerRadius={105} label>{item.data.map((_:any,i:number)=><Cell key={i} fill={['#5978e8','#36a38b','#e6a34b','#b17de0','#e56d82'][i%5]}/>)}</Pie></PieChart>
:item.chart_type==='scatter'?<ScatterChart><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" type="number" label={{value:item.x_label||'X',position:'insideBottom',offset:-5}}/><YAxis dataKey="value" label={{value:item.y_label||'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Scatter data={item.data} fill="#5978e8"/></ScatterChart>
:item.chart_type==='radar'?<RadarChart data={item.data}><PolarGrid/><PolarAngleAxis dataKey="x"/><PolarRadiusAxis/><Tooltip/><Radar dataKey="value" stroke="#5978e8" fill="#5978e8" fillOpacity={.25}/></RadarChart>
:item.chart_type==='box'?<ComposedChart data={item.data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label||'Category',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label||'Value',angle:-90,position:'insideLeft'}}/><Tooltip/><Bar dataKey="median" fill="#5978e8"><ErrorBar dataKey="range" direction="y" width={5}/></Bar></ComposedChart>
:item.chart_type==='density'?<AreaChart data={item.data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" type="number" label={{value:item.x_label||'Value',position:'insideBottom',offset:-5}}/><YAxis label={{value:'Density',angle:-90,position:'insideLeft'}}/><Tooltip/><Area dataKey="value" stroke="#36a38b" fill="#36a38b" fillOpacity={.2}/></AreaChart>
:item.chart_type==='line'?<LineChart data={item.data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label||'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label||'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Line dataKey="value" stroke="#5978e8"/></LineChart>
:item.chart_type==='area'?<AreaChart data={item.data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label||'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label||'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Area dataKey="value" stroke="#5978e8" fill="#5978e8" fillOpacity={.2}/></AreaChart>
:<BarChart data={item.data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label||'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label||'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Bar dataKey="value" fill="#5978e8"/></BarChart>}
</ResponsiveContainer>}
{item.chart_type!=='heatmap'&&<details className="table-disclosure"><summary>View chart data · {(item.data||[]).length.toLocaleString()} rows</summary>{tableContent(item.data||[])}</details>}
</section>)}
</div>}
</div>}
{p&&tab==='train'&&<div className="stack">{card('Step 1 · Configure model',<><div className="formgrid">{field('Target',<select value={target} onChange={e=>{STG(e.target.value);SF(f=>f.filter(c=>c!==e.target.value))}}><option value="">Choose target</option>{targetColumns.map((c:any)=><option key={c.column} value={c.column}>{c.column} ({c.dtype})</option>)}</select>)}{field('Task',<select value={task} onChange={e=>{const next=e.target.value;STK(next);SNModel(next==='classification'?'Logistic Regression':'Ridge Regression');const eligible=(p.columns_info||[]).filter((c:any)=>c.unique>=2&&(next==='regression'?c.kind==='numeric':c.unique<=100));if(!eligible.some((c:any)=>c.column===target))STG(eligible[0]?.column||'')}}><option value="classification">Classification</option><option value="regression">Regression</option></select>)}{field('Model selection',<select value={mode} onChange={e=>SM(e.target.value)}><option value="auto">Select all algorithms</option><option value="manual">Choose multiple algorithms</option></select>)}{mode==='manual'&&<div className="model-picker"><label className="select-all-models"><input type="checkbox" checked={selectedModels.length===models.length} onChange={e=>SSelectedModels(e.target.checked?[...models]:[])}/>Select all models ({models.length})</label><div className="checkgrid">{models.map(m=><label key={m}><input type="checkbox" checked={selectedModels.includes(m)} onChange={e=>SSelectedModels(current=>e.target.checked?[...current,m]:current.filter(name=>name!==m))}/>{m}</label>)}</div><small>{selectedModels.length} of {models.length} models selected</small></div>}{field('Test split '+Math.round(split*100)+'%',<input type="range" min=".1" max=".5" step=".05" value={split} onChange={e=>SSplit(+e.target.value)}/>)}{field('Feature selection',<select value={selection} onChange={e=>SSEL(e.target.value)}><option value="none">None</option><option value="kbest">Select K best</option><option value="variance">Remove constant features</option></select>)}{selection==='kbest'&&field('K features '+k,<input type="range" min="1" max="50" value={k} onChange={e=>SKN(+e.target.value)}/>)}{field('PCA '+pcaK+' components',<><input type="checkbox" checked={pca} onChange={e=>SPCA(e.target.checked)}/><input type="range" min="1" max="20" value={pcaK} onChange={e=>SPK(+e.target.value)} disabled={!pca}/></>)}</div><label><input type="checkbox" checked={scale} onChange={e=>SSC(e.target.checked)}/> Scale numeric features</label> <label><input type="checkbox" checked={drop} onChange={e=>SD(e.target.checked)}/> Drop columns with more than 80% missing</label><h3>Input features</h3><p>{task==='classification'?'Classification predicts a category or class label.':'Regression predicts a continuous numeric value.'} Select at least one input feature. The target is excluded automatically.</p><div className="checkgrid">{p.columns_info.filter((c:any)=>c.column!==target).map((c:any)=><label key={c.column}><input type="checkbox" checked={features.includes(c.column)} onChange={e=>SF(a=>e.target.checked?[...a,c.column]:a.filter(v=>v!==c.column))}/>{c.column}</label>)}</div><p>Available {task} algorithms: {models.join(', ')}.</p><button className={hasActiveTraining?'stop-button':'primary'} onClick={()=>activeTrainingJob?void cancelTraining(activeTrainingJob.job_id):void train()} disabled={busy||Boolean(activeTrainingJob?.cancel_requested)||(!hasActiveTraining&&(!target||!features.some(c=>c!==target)||(mode==='manual'&&selectedModels.length===0)))}>{hasActiveTraining?(activeTrainingJob?.cancel_requested?<>Stopping…</>:<><span className="stop-square"/>Stop training</>) : <> <FlaskConical size={15}/>Train {mode==='auto'?'and compare all '+task+' models':selectedModels.length+' selected models'}</>}</button></>)}{trainingQueue.length>0&&card('Training queue',<div className="stack">{trainingQueue.map((j:any)=><div className="savedrow" key={j.job_id}><span>{j.model_name} · {j.task} · {j.target}<small>{j.status==='queued'?'Waiting in queue':j.status==='running'?'Training now':j.status==='completed'?'Completed':j.error||'Failed'}{j.queue_position? ' · Queue position '+j.queue_position:''}</small></span><span className="queue-status">{j.status}</span>{(j.status==='queued'||j.status==='running')&&<button className="stop-button compact-stop" onClick={()=>void cancelTraining(j.job_id)}><span className="stop-square"/>Stop</button>}</div>)}</div>)}</div>}
 {p && tab === 'evaluation' && (
  <div className="stack" ref={evaluationRef}>
    {!model ? (
      card('No model trained', <button className="primary" onClick={() => S('train')}>Configure training</button>)
    ) : (
      <>
        <div className="metrics">
          {metric('Best model', model.best_model)}
          {metric('Training rows', model.train_rows)}
          {metric('Test rows', model.test_rows)}
          {metric('Target', model.target)}
        </div>
        {card('Model leaderboard', (
          <>
            {table(model.leaderboard)}
            {model.warnings?.map((warning: string, index: number) => <p key={index} className="hint">{warning}</p>)}
            <button className="secondary" onClick={() => download('/api/models/' + model.model_id + '/download')}>
              <Download size={16}/> Download model
            </button>
          </>
        ))}
        {card('Evaluation metrics', (
          <div className="metrics">
            {Object.entries(model.best_evaluation?.metrics || {}).map(([key, value]) => metric(key, fmt(value)))}
          </div>
        ))}
        {model.task === 'classification' && model.best_evaluation?.confusion_matrix && card(
          'Confusion matrix · rows are actual classes, columns are predicted classes',
          table(confusionRows(model.best_evaluation.confusion_matrix))
        )}
        {card(
          model.task === 'classification' ? 'Actual vs predicted class labels' : 'Actual vs predicted values',
          model.task === 'classification' ? (
            <>
              <p>Each row represents one test example. Correct shows whether the predicted class matches the actual class.</p>
              {table(actualPredictedRows)}
            </>
          ) : (
            <>
              <p>Each point represents one test example. Horizontal axis: actual value. Vertical axis: predicted value.</p>
              {regressionPoints.length > 0 ? (
                <ResponsiveContainer width="100%" height={340}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis type="number" dataKey="actual" name="Actual"/>
                    <YAxis type="number" dataKey="predicted" name="Predicted"/>
                    <Tooltip/>
                    <Legend/>
                    <Scatter name="Predictions" data={regressionPoints} fill="#5978e8"/>
                  </ScatterChart>
                </ResponsiveContainer>
              ) : <p>No finite numeric prediction pairs are available.</p>}
              {table(actualPredictedRows)}
            </>
          )
        )}
      </>
    )}
  </div>
 )}
 {p&&tab==='predict'&&<div className="stack">{card('External CSV prediction',!model?<button className="primary" onClick={()=>S('train')}>Train a model first</button>:<><p>Upload a test CSV with the same feature columns.</p><label className="upload"><UploadCloud/> Choose test CSV<input type="file" accept=".csv" onChange={e=>predict(e.target.files?.[0])}/></label>{prediction&&<><div className="metrics">{metric('Rows predicted',prediction.rows)}{metric('Output column',prediction.predicted_column)}</div>{table(prediction.preview)}<button className="primary" onClick={()=>download(prediction.download_url)}>Download predictions CSV</button></>}</>)}</div>}
 {p&&<div className="phase-nav"><button className="phase-back" onClick={()=>goToPhase(Math.max(0,phaseIndex-1))} disabled={phaseIndex===0}><ChevronLeft size={16}/>Previous phase</button><div className="phase-status"><small>WORKFLOW PHASE {phaseIndex+1} OF {nav.length}</small><strong>{nav[phaseIndex]?.[1]}</strong><div className="phase-progress">{nav.map((item:any,i:number)=><span key={item[0]} className={i<=phaseIndex?'complete':''}/>)}</div></div><button className="primary phase-next" onClick={()=>goToPhase(phaseIndex===nav.length-1?0:phaseIndex+1)}>{phaseIndex===nav.length-1?'Back to overview':'Next phase'}<ChevronRight size={16}/></button></div>}
 <footer>AutoML Studio · React + FastAPI · Local-first</footer></section></main></div>
}
function App(){
 const [user,setUser]=useState<{id:string;email:string}|null>(null)
 const [ready,setReady]=useState(false)
 useEffect(()=>{(async()=>{try{let result=await authClient.getSession();if(!result.error&&result.data?.user&&result.data.session?.token){setUser(result.data.user);return}result=await authClient.getSession({fetchOptions:{headers:{'X-Force-Fetch':'true'}}});if(result.error)throw new Error(result.error.message);if(result.data?.user&&result.data.session?.token){setUser(result.data.user);return}/* no active session, stay on login */}catch(error){console.error('Could not restore the sign-in session:',error)}finally{setReady(true)}})()},[])
 async function signOut(){try{await authClient.signOut()}catch{}setUser(null)}
 if(!ready)return <main className="auth-page"><section className="auth-card" style={{textAlign:'center'}}><p>Checking your session…</p></section></main>
 return user?<Workspace key={user.id} user={user} onSignOut={signOut}/>:<AuthScreen onAuthenticated={setUser}/>
}
export default App
