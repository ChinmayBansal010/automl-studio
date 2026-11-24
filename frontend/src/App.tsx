import { useEffect, useRef, useState } from 'react'
import { Activity, BarChart3, BrainCircuit, ChevronLeft, ChevronRight, Database, Download, FilePlus2, FlaskConical, Gauge, History, House, LogOut, RefreshCw, RotateCcw, Sparkles, Target, UploadCloud } from 'lucide-react'
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { AuthScreen } from './AuthScreen'
import { PublicPage } from './PublicPage'
import { SystemPage } from './SystemPage'
import { ChartCard } from './components/ChartCard'
import { WorkspaceHeader } from './components/WorkspaceHeader'
import { WorkspaceSidebar } from './components/WorkspaceSidebar'
import { DashboardHome } from './DashboardHome'
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
  const request=indexedDB.open(BACKUP_DB,3)
  request.onupgradeneeded=()=>{const db=request.result;if(!db.objectStoreNames.contains('datasets'))db.createObjectStore('datasets',{keyPath:'key'});if(!db.objectStoreNames.contains('experiments'))db.createObjectStore('experiments',{keyPath:'key'});if(!db.objectStoreNames.contains('datasetVersions'))db.createObjectStore('datasetVersions',{keyPath:'key'})}
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
async function saveDatasetVersionBackup(userId:string,datasetId:string,filename:string,reason:string,blob:Blob,rows:number,columns:number):Promise<void>{
 const db=await openBackupDb()
 const versionId=crypto.randomUUID()
 const key=userId+':'+datasetId+':'+versionId
 await new Promise<void>((resolve,reject)=>{const tx=db.transaction('datasetVersions','readwrite');const store=tx.objectStore('datasetVersions');store.put({key,userId,datasetId,version_id:versionId,filename,reason,blob,rows,columns,size_bytes:blob.size,created_at:new Date().toISOString()});tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)})
 const all=await new Promise<any[]>((resolve,reject)=>{const request=db.transaction('datasetVersions','readonly').objectStore('datasetVersions').getAll();request.onsuccess=()=>resolve(request.result||[]);request.onerror=()=>reject(request.error)})
 const stale=all.filter(item=>item.userId===userId&&item.datasetId===datasetId).sort((a,b)=>String(b.created_at).localeCompare(String(a.created_at))).slice(25)
 if(stale.length){const tx=db.transaction('datasetVersions','readwrite');for(const item of stale)tx.objectStore('datasetVersions').delete(item.key);await new Promise<void>((resolve,reject)=>{tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)})}
 db.close()
}
async function readDatasetVersionBackups(userId:string,datasetId:string):Promise<any[]>{
 const db=await openBackupDb()
 const all=await new Promise<any[]>((resolve,reject)=>{const request=db.transaction('datasetVersions','readonly').objectStore('datasetVersions').getAll();request.onsuccess=()=>resolve(request.result||[]);request.onerror=()=>reject(request.error)})
 db.close()
 return all.filter(item=>item.userId===userId&&item.datasetId===datasetId).sort((a,b)=>String(b.created_at).localeCompare(String(a.created_at)))
}
async function saveExperimentHistory(userId:string,jobs:any[]):Promise<void>{
 const db=await openBackupDb()
 const safeJobs=jobs.map(({result,...job})=>job)
 await new Promise<void>((resolve,reject)=>{const tx=db.transaction('experiments','readwrite');tx.objectStore('experiments').put({key:userId,jobs:safeJobs,updatedAt:new Date().toISOString()});tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)})
 db.close()
}
async function readExperimentHistory(userId:string):Promise<any[]>{
 const db=await openBackupDb()
 const result=await new Promise<any[]>((resolve,reject)=>{const request=db.transaction('experiments','readonly').objectStore('experiments').get(userId);request.onsuccess=()=>resolve(request.result?.jobs||[]);request.onerror=()=>reject(request.error)})
 db.close()
 return result
}
function setActiveDatasetRef(userId:string,datasetId:string,filename:string):void{try{localStorage.setItem('automl-active-dataset:'+userId,JSON.stringify({datasetId,filename}))}catch{}}
function getActiveDatasetRef(userId:string):{datasetId:string;filename:string}|null{try{const value=localStorage.getItem('automl-active-dataset:'+userId);return value?JSON.parse(value):null}catch{return null}}
const fmt=(v:any)=>v===null||v===undefined?'—':typeof v==='number'?Number(v.toFixed?.(4)??v).toLocaleString():String(v)
const durationLabel=(seconds:any)=>seconds===null||seconds===undefined?'Calculating…':seconds<60?`${seconds}s remaining`:`${Math.floor(seconds/60)}m ${seconds%60}s remaining`
function Workspace({user,onSignOut,initialTab,onNavigate}:{user:{id:string;email:string};onSignOut:()=>Promise<void>;initialTab:string;onNavigate:(tab:string)=>void}){
 const uploadInput=useRef<HTMLInputElement>(null)
 const dashboardReady=useRef(false)
 const evaluationRef=useRef<HTMLDivElement>(null)
 const [initialLoading,setInitialLoading]=useState(true)
 const [tab,setTab]=useState(initialTab),[p,SP]=useState<any>(null),[eda,SE]=useState<any>(null),[err,ER]=useState(''),[busy,SB]=useState(false),[ws,SW]=useState<any>({saved_datasets:[],saved_models:0}),[jobs,SJ]=useState<any[]>([])
 const [op,SO]=useState('fill_missing'),[strategy,SS]=useState('median'),[cols,SC]=useState<string[]>([]),[featureCols,SFeatureCols]=useState<string[]>([]),[threshold,ST]=useState(.5),[newName,SN]=useState(''),[value,SV]=useState(''),[components,SCM]=useState(2),[sampleSize,SSample]=useState(1000)
 const [kind,SK]=useState('bar'),[x,SX]=useState(''),[y,SY]=useState(''),[agg,SA]=useState('mean'),[rows,SR]=useState(20),[grid,SG]=useState(1),[chart,SCH]=useState<any>(null)
 const S=(nextTab:string)=>{setTab(nextTab);onNavigate(nextTab)}
 useEffect(()=>{setTab(initialTab)},[initialTab])
 const [target,STG]=useState(''),[task,STK]=useState('classification'),[mode,SM]=useState('auto'),[modelName,SNModel]=useState('Random Forest'),[selectedModels,SSelectedModels]=useState<string[]>([]),[split,SSplit]=useState(.2),[selection,SSEL]=useState('none'),[k,SKN]=useState(10),[pca,SPCA]=useState(false),[pcaK,SPK]=useState(3),[scale,SSC]=useState(true),[drop,SD]=useState(false),[features,SF]=useState<string[]>([]),[model,SModel]=useState<any>(null),[prediction,SPred]=useState<any>(null),[trainingQueue,STQ]=useState<any[]>([]),[datasetVersions,SVersions]=useState<any[]>([])
 const models=task==='classification'?['Logistic Regression','Decision Tree','Random Forest','Extra Trees','Gradient Boosting','Histogram Gradient Boosting','AdaBoost','K-Nearest Neighbors','Support Vector Machine','Naive Bayes','Neural Network (MLP)']:['Ridge Regression','Decision Tree','Random Forest','Extra Trees','Gradient Boosting','Histogram Gradient Boosting','AdaBoost','K-Nearest Neighbors','Support Vector Regression','Neural Network (MLP)']
 const targetColumns=(p?.columns_info||[]).filter((c:any)=>c.unique>=2&&(task==='regression'?c.kind==='numeric':c.unique<=100))
 const chartXColumns=(p?.columns_info||[]).filter((c:any)=>kind==='heatmap'?false:['histogram','scatter','density','ecdf','qq'].includes(kind)?c.kind==='numeric':true)
 const chartYColumns=(p?.columns_info||[]).filter((c:any)=>['line','area','step','scatter','bar','box','radar'].includes(kind)?c.kind==='numeric':false)
 function saveCanvas(canvas:HTMLCanvasElement,filename:string){
  const link=document.createElement('a');link.download=filename;link.href=canvas.toDataURL('image/png');link.click()
 }
 function chartExportName(index:number){return ((p?.filename||'automl-dashboard').replace(/\.[^.]+$/,'').replace(/[^a-z0-9-_]+/gi,'_')+'_chart_'+(index+1)+'.png')}
 async function downloadChartPng(index:number){
  const item=Array.isArray(chart)?chart[index]:null
  const cardNode=document.querySelectorAll<HTMLElement>('.chart-card')[index]
  if(!item||!cardNode){ER('Could not find this graph to export.');return}
  SB(true);ER('')
  try{
   const canvas=document.createElement('canvas')
   if(item.chart_type==='heatmap'){
    const rows=Array.isArray(item.data)?item.data:[]
    const cols=rows.length?Object.keys(rows[0]):[]
    const cellWidth=120,rowHeight=36,padding=24
    canvas.width=Math.max(600,padding*2+cols.length*cellWidth);canvas.height=padding*2+(rows.length+1)*rowHeight
    const ctx=canvas.getContext('2d')
    if(!ctx)throw new Error('Canvas is unavailable')
    ctx.fillStyle='#ffffff';ctx.fillRect(0,0,canvas.width,canvas.height)
    ctx.font='14px Arial';ctx.textBaseline='middle'
    cols.forEach((col:string,i:number)=>{ctx.fillStyle='#263247';ctx.fillText(col,padding+i*cellWidth+8,padding+rowHeight/2)})
    rows.forEach((row:any,r:number)=>cols.forEach((col:string,c:number)=>{const x=padding+c*cellWidth,y=padding+(r+1)*rowHeight;ctx.fillStyle=r%2?'#f4f6fa':'#ffffff';ctx.fillRect(x,y,cellWidth,rowHeight);ctx.strokeStyle='#dce2eb';ctx.strokeRect(x,y,cellWidth,rowHeight);ctx.fillStyle='#263247';ctx.fillText(String(row[col]??''),x+8,y+rowHeight/2)}))
   }else{
    const svg=cardNode.querySelector('svg')
    if(!svg)throw new Error('The graph image is not available yet')
    const bounds=svg.getBoundingClientRect(),clone=svg.cloneNode(true) as SVGSVGElement
    clone.setAttribute('xmlns','http://www.w3.org/2000/svg');clone.setAttribute('width',String(Math.max(1,Math.round(bounds.width*2))));clone.setAttribute('height',String(Math.max(1,Math.round(bounds.height*2))))
    const source=new XMLSerializer().serializeToString(clone)
    const url=URL.createObjectURL(new Blob([source],{type:'image/svg+xml;charset=utf-8'}))
    try{
     const image=new Image()
     await new Promise<void>((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error('Could not render graph image'));image.src=url})
     canvas.width=Math.max(1,Math.round(bounds.width*2));canvas.height=Math.max(1,Math.round(bounds.height*2))
     const ctx=canvas.getContext('2d');if(!ctx)throw new Error('Canvas is unavailable')
     ctx.fillStyle='#ffffff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height)
    }finally{URL.revokeObjectURL(url)}
   }
   saveCanvas(canvas,chartExportName(index))
  }catch(error){ER('Graph export failed: '+String(error))}
  finally{SB(false)}
 }
 function downloadDashboardPdf(){
  if(!Array.isArray(chart)||chart.length===0){ER('There are no graphs to export.');return}
  const previousTitle=document.title
  document.title=((p?.filename||'automl-dashboard').replace(/\.[^.]+$/,'')+'_dashboard')
  const restore=()=>{document.title=previousTitle;window.removeEventListener('afterprint',restore)}
  window.addEventListener('afterprint',restore)
  window.print()
  window.setTimeout(restore,1000)
 }

 const activeTrainingJob=trainingQueue.find((j:any)=>j.status==='queued'||j.status==='running')
 const hasActiveTraining=Boolean(activeTrainingJob)
 const prepColumns=(p?.columns_info||[]).filter((c:any)=>((op==='drop_outliers'||op==='clip_outliers'||(op==='fill_missing'&&['mean','median','zero'].includes(strategy)))?c.kind==='numeric':true))
 const selectedNumericCount=featureCols.filter((c:string)=>(p?.columns_info||[]).some((info:any)=>info.column===c&&info.kind==='numeric')).length
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

 async function refresh(){const a=await api('/workspace'),b=await api('/workspace/jobs');const serverJobs=b.jobs||[];const localJobs=await readExperimentHistory(user.id).catch(()=>[]);const byId=new Map<string,any>();for(const job of localJobs)byId.set(job.job_id,{...job,model_available:false,can_retry:Boolean(job.request_config)});for(const job of serverJobs)byId.set(job.job_id,job);const merged=[...byId.values()].sort((x,y)=>String(y.created_at||'').localeCompare(String(x.created_at||'')));SW(a);SJ(merged);STQ(merged);void saveExperimentHistory(user.id,merged).catch(()=>{})}
 useEffect(()=>{if(!p?.dataset_id)return;dashboardReady.current=false;try{const saved=localStorage.getItem('automl-dashboard:'+user.id+':'+p.dataset_id);if(saved){const value=JSON.parse(saved);SCH(Array.isArray(value.charts)?value.charts:[]);SG([1,2,3].includes(value.columns)?value.columns:1)}else SCH([])}catch{SCH([])}dashboardReady.current=true},[p?.dataset_id,user.id])
 useEffect(()=>{if(!dashboardReady.current||!p?.dataset_id)return;try{localStorage.setItem('automl-dashboard:'+user.id+':'+p.dataset_id,JSON.stringify({charts:Array.isArray(chart)?chart:[],columns:grid}))}catch{}},[chart,grid,p?.dataset_id,user.id])
 useEffect(()=>{if(trainingQueue.length)void saveExperimentHistory(user.id,trainingQueue).catch(()=>{})},[trainingQueue,user.id])
 useEffect(()=>{(async()=>{await refresh();await recoverActiveDataset()})().catch(error=>ER(error instanceof Error?error.message:String(error))).finally(()=>setInitialLoading(false))},[])
 async function signOut(){try{await onSignOut()}catch(error){ER(String(error))}}
 async function cacheDatasetBackup(datasetId:string,filename:string){
  try{const token=await getValidToken();const response=await fetchWithRetry(API+'/datasets/'+datasetId+'/download',{headers:{Authorization:'Bearer '+token}});if(response.ok)await saveDatasetBackup(user.id,datasetId,filename,await response.blob())}catch{}
 }
 async function loaded(d:any){ER('');SP(d);setActiveDatasetRef(user.id,d.dataset_id,d.filename);SC(d.columns_info.map((c:any)=>c.column));SFeatureCols([]);SF(d.columns_info.map((c:any)=>c.column));SX(d.columns_info[0]?.column||'');SY(d.types?.numeric?.[0]||d.columns_info[1]?.column||'');STG('');SModel(null);SPred(null);await cacheDatasetBackup(d.dataset_id,d.filename);const backup=await readDatasetBackup(user.id,d.dataset_id);if(backup)await saveDatasetVersionBackup(user.id,d.dataset_id,d.filename,d.operation?'Updated: '+d.operation:'Dataset snapshot',backup.blob,d.rows||0,d.columns||0).catch(()=>{});SE(await api('/datasets/'+d.dataset_id+'/eda'));try{const versions=await api('/datasets/'+d.dataset_id+'/versions');const local=await readDatasetVersionBackups(user.id,d.dataset_id);const ids=new Set((versions.versions||[]).map((v:any)=>v.version_id));SVersions([...(versions.versions||[]),...local.filter(v=>!ids.has(v.version_id)).map(({blob,key,userId,datasetId,...v}:any)=>v)].sort((a:any,b:any)=>String(b.created_at).localeCompare(String(a.created_at))))}catch{const local=await readDatasetVersionBackups(user.id,d.dataset_id);SVersions(local.map(({blob,key,userId,datasetId,...v}:any)=>v))}await refresh()}
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
 async function downloadProcessedCsv(){if(!p)return;try{await ensureDatasetReady();await download('/api/datasets/'+p.dataset_id+'/download')}catch(e){ER(e instanceof Error?e.message:String(e))}}
 async function upload(file?:File){if(!file)return;SB(true);ER('');try{const f=new FormData();f.append('file',file);const dataset=await api('/datasets',{method:'POST',body:f});try{await saveDatasetBackup(user.id,dataset.dataset_id,file.name,file)}catch{}setActiveDatasetRef(user.id,dataset.dataset_id,file.name);await loaded(dataset);S('overview')}catch(e){ER(String(e))}finally{SB(false)}}
 async function restore(id:string){SB(true);try{await loaded(await api('/workspace/datasets/'+id+'/load',{method:'POST'}));S('overview')}catch(e){ER(String(e))}finally{SB(false)}}
 async function prep(){if(!p)return;SB(true);ER('');try{await ensureDatasetReady();const r=await api('/datasets/'+p.dataset_id+'/preprocess',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation:op,columns:cols,strategy,value,threshold,new_name:newName,sample_size:sampleSize})});await loaded({...r,filename:p.filename})}catch(e){ER(String(e))}finally{SB(false)}}
 async function feat(operation:string){if(!p)return;const numeric=featureCols.filter(c=>(p.columns_info||[]).some((info:any)=>info.column===c&&info.kind==='numeric'));if(operation==='pca'&&numeric.length<2){ER('Select at least two numeric columns for PCA.');return}if(operation!=='pca'&&featureCols.length===0){ER('Select at least one column for feature engineering.');return}if(['standardize','minmax','log_transform','polynomial'].includes(operation)&&numeric.length===0){ER('Select at least one numeric column for this transformation.');return}SB(true);ER('');try{await ensureDatasetReady();const r=await api('/datasets/'+p.dataset_id+'/feature-engineering',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,columns:featureCols,components})});await loaded({...r,filename:p.filename})}catch(e){ER(e instanceof Error?e.message:String(e))}finally{SB(false)}}
 async function makeChart(){if(!p)return;if(kind!=='heatmap'&&!x){ER('Select a valid X-axis column.');return}if(['line','area','step','scatter','box','radar'].includes(kind)&&!y){ER('Select a valid numeric Y-axis column.');return}SB(true);ER('');try{await ensureDatasetReady();const generated=await api('/datasets/'+p.dataset_id+'/chart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({chart_type:kind,x:kind==='heatmap'?null:x,y:['pie','count','histogram','heatmap','density','ecdf','qq'].includes(kind)?null:y,aggregation:agg,rows,columns:grid})});SCH((current:any)=>[...(Array.isArray(current)?current:current?[current]:[]),generated])}catch(e){ER(String(e))}finally{SB(false)}}
 async function watchTraining(jobId:string){try{while(true){await new Promise(resolve=>setTimeout(resolve,2000));const status=await api('/training-jobs/'+jobId);STQ(q=>q.map(j=>j.job_id===jobId?{...j,...status}:j));if(status.status==='completed'){if(status.result){SModel(status.result);S('evaluation');window.setTimeout(()=>evaluationRef.current?.scrollIntoView({behavior:'smooth',block:'start'}),80)}await refresh();break}if(status.status==='failed'){ER(status.error||'Training failed.');await refresh();break}if(status.status==='cancelled'){STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:'cancelled'}:j));await refresh();break}}}catch(e){const detail=e instanceof Error?e.message:String(e);STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:'interrupted',error:detail,can_retry:true}:j));try{await recoverActiveDataset()}catch{}try{await refresh()}catch{}ER('The training service became unavailable before it could return a result. Your experiment configuration is retained in history when the server has persisted it. Retry will restart the experiment from the beginning.')} }
 async function train(){if(!p||!target){ER('Select a valid target column.');return}if(!features.some(c=>c!==target)){ER('Select at least one input feature.');return}SB(true);ER('');try{await ensureDatasetReady();const queued=await api('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:p.dataset_id,target,task,mode,model_name:modelName,model_names:mode==='manual'?selectedModels:undefined,test_size:split,scale_numeric:scale,drop_high_missing:drop,selected_features:features.filter(c=>c!==target),feature_selection:selection,k_features:k,use_pca:pca,pca_components:pcaK})});STQ(q=>[{job_id:queued.job_id,status:queued.status,queue_position:queued.queue_position,target,task,model_name:mode==='auto'?'All '+task+' models':selectedModels.join(', '),created_at:new Date().toISOString(),request_config:{dataset_id:p.dataset_id,target,task,mode,model_name:modelName,model_names:mode==='manual'?selectedModels:undefined,test_size:split,scale_numeric:scale,drop_high_missing:drop,selected_features:features.filter(c=>c!==target),feature_selection:selection,k_features:k,use_pca:pca,pca_components:pcaK}},...q].slice(0,10));void watchTraining(queued.job_id)}catch(e){ER(String(e))}finally{SB(false)}}
 async function cancelTraining(jobId:string){try{const result=await api('/training-jobs/'+jobId+'/cancel',{method:'POST'});STQ(q=>q.map(j=>j.job_id===jobId?{...j,status:result.status==='cancelled'?'cancelled':'running',cancel_requested:result.status==='cancelling',message:result.message}:j))}catch(e){ER(e instanceof Error?e.message:String(e))}}
 async function skipCurrentModel(jobId:string){try{const result=await api('/training-jobs/'+jobId+'/skip-current',{method:'POST'});STQ(q=>q.map(j=>j.job_id===jobId?{...j,skip_requested_model:result.model_name,message:result.message}:j))}catch(e){ER(e instanceof Error?e.message:String(e))}}
 async function retryTraining(jobId:string){try{const previous=trainingQueue.find((job:any)=>job.job_id===jobId);const datasetId=previous?.request_config?.dataset_id;if(datasetId){try{await api('/datasets/'+datasetId+'/profile')}catch{const backup=await readDatasetBackup(user.id,datasetId);if(!backup)throw new Error('The dataset needed for this experiment is not available. Restore or upload its CSV before retrying.');const form=new FormData();form.append('file',backup.blob,backup.filename);await api('/datasets/'+datasetId+'/restore',{method:'POST',body:form})}}const result=await api('/training-jobs/'+jobId+'/retry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(previous?.request_config||{})});STQ(q=>[{job_id:result.job_id,status:'queued',queue_position:result.queue_position,retry_of:jobId,created_at:new Date().toISOString(),model_name:'Retry experiment'},...q]);void watchTraining(result.job_id);S('train')}catch(e){ER(e instanceof Error?e.message:String(e))}}
 async function restoreVersion(versionId:string){if(!p)return;SB(true);try{let restored:any;try{restored=await api('/datasets/'+p.dataset_id+'/versions/'+versionId+'/restore',{method:'POST'})}catch(error){const local=await readDatasetVersionBackups(user.id,p.dataset_id);const version=local.find(item=>item.version_id===versionId);if(!version)throw error;const form=new FormData();form.append('file',version.blob,version.filename);restored=await api('/datasets/'+p.dataset_id+'/restore',{method:'POST',body:form})}await loaded({...restored,filename:p.filename,operation:'Restored dataset version'});S('prepare')}catch(e){ER(e instanceof Error?e.message:String(e))}finally{SB(false)}}
 async function predict(file?:File){if(!file||!model)return;SB(true);try{const f=new FormData();f.append('file',file);f.append('model_id',model.model_id);SPred(await api('/predict',{method:'POST',body:f}))}catch(e){ER(String(e))}finally{SB(false)}}
 async function download(url:string,filename?:string){try{if(!BACKEND_URL)throw new Error('Backend URL is not configured. Set VITE_BACKEND_URL for this deployment.');const token=await getValidToken();const response=await fetchWithRetry(url.startsWith('/')?BACKEND_URL+url:API+url,{headers:{Authorization:'Bearer '+token}});if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(body.detail||('Download failed '+response.status))}const disposition=response.headers.get('content-disposition')||'';const match=disposition.match(/filename\*?=(?:UTF-8''|")?([^";]+)/i);const serverFilename=match?.[1]?decodeURIComponent(match[1].replace(/"/g,'').trim()):undefined;const objectUrl=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=objectUrl;a.download=filename||serverFilename||'download';document.body.appendChild(a);a.click();a.remove();window.setTimeout(()=>URL.revokeObjectURL(objectUrl),1000)}catch(e){ER(e instanceof Error?e.message:String(e))}}
 function downloadEvaluationResults(data:any){try{const records:any[]=[];for(const item of data.leaderboard||[])records.push({record_type:'leaderboard',...item});const metrics=data.best_evaluation?.metrics||{};for(const [metric,value] of Object.entries(metrics))records.push({record_type:'best_model_metric',metric,value});const best=data.results?.find((item:any)=>item.model_name===data.best_model)||data.results?.[0]||{};const actual=best.actual||data.actual||[],predictions=best.predictions||data.predictions||[];for(let i=0;i<Math.min(actual.length,predictions.length);i++)records.push({record_type:'test_prediction',row_index:i,actual:actual[i],predicted:predictions[i],correct:data.task==='classification'?actual[i]===predictions[i]:''});const confusion=data.best_evaluation?.confusion_matrix||{};for(let i=0;i<(confusion.labels||[]).length;i++)for(let j=0;j<(confusion.labels||[]).length;j++)records.push({record_type:'confusion_matrix',actual_class:confusion.labels[i],predicted_class:confusion.labels[j],count:confusion.matrix?.[i]?.[j]??0});if(!records.length)throw new Error('No evaluation results are available for this experiment.');const headers=Array.from(new Set(records.flatMap(row=>Object.keys(row))));const escape=(value:any)=>{const text=value===null||value===undefined?'':String(value);return /[",\n\r]/.test(text)?'"'+text.replace(/"/g,'""')+'"':text};const csv='\uFEFF'+[headers.map(escape).join(','),...records.map(row=>headers.map(key=>escape(row[key])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='automl-evaluation-'+String(data.model_id||'results').slice(0,8)+'.csv';document.body.appendChild(link);link.click();link.remove();window.setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){ER(e instanceof Error?e.message:String(e))}}
 const nav=[['home','Home',House],['overview','Data overview',Gauge],['prepare','Data preparation',RefreshCw],['charts','Visual analysis',BarChart3],['train','Model training',FlaskConical],['evaluation','Model evaluation',Activity],['predict','Predictions',Target],['models','Model library',BrainCircuit],['experiments','Run history',History]] as any[]
 const workflowNav=nav.filter(t=>t[0]!=='home').slice(0,6)
 const phaseIndex=Math.max(0,workflowNav.findIndex(t=>t[0]===tab))
 const goToPhase=(index:number)=>S(workflowNav[index]?.[0]||'overview')
 const card=(title:string,body:any)=><section className="card"><h2>{title}</h2>{body}</section>
 const field=(label:string,child:any)=><label className="field"><span>{label}</span>{child}</label>
 const metric=(label:string,val:any)=><div className="metric"><small>{label}</small><strong>{fmt(val)}</strong></div>
 const tableContent=(data:any[])=> <div className="tablewrap"><table><thead><tr>{Object.keys(data[0]).map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>{data.slice(0,100).map((r:any,i:number)=><tr key={i}>{Object.keys(data[0]).map(c=><td key={c}>{r[c]===null||r[c]===undefined?'—':String(r[c])}</td>)}</tr>)}</tbody></table></div>
 const table=(data:any[])=>!data?.length?<p>No rows to display.</p>:data.length>12?<details className="table-disclosure"><summary>Show table · {data.length.toLocaleString()} rows (first 100 displayed)</summary>{tableContent(data)}</details>:tableContent(data)
 return <div className="app"><WorkspaceSidebar items={nav} activeTab={tab} email={user.email} onNavigate={(nextTab:string)=>S(nextTab)} onSignOut={signOut}/><main><WorkspaceHeader title={nav.find(t=>t[0]===tab)?.[1]||'Workspace'} datasetName={p?.filename} disabled={busy||hasActiveTraining} uploadInputRef={uploadInput} onFileChange={e=>{const file=e.target.files?.[0];if(file)void upload(file);e.currentTarget.value=''}}/><section className="content">
 {err&&<div className="error">{err}<button onClick={()=>ER('')}>Dismiss</button></div>}{busy&&<div className="busy-panel"><span className="loader-ring"/><div><strong>Working on your request</strong><small>Processing data and updating the workspace…</small></div><span className="busy-pulse"/></div>}{initialLoading&&!p&&<div className="skeleton-dashboard" aria-label="Loading workspace"><div className="skeleton-metrics">{[1,2,3,4].map(i=><div className="skeleton skeleton-metric" key={i}/>)}</div><div className="skeleton skeleton-card"/><div className="skeleton skeleton-card short"/></div>}{p&&tab!=='overview'&&tab!=='home'&&<div className="dataset-context"><div><small>ACTIVE DATASET · LIVE PROFILE</small><strong>{p.filename}</strong><span>{p.rows??p.shape?.[0]??'—'} rows · {p.columns??p.shape?.[1]??p.columns_info?.length??'—'} columns · {p.missing_total??0} missing cells</span></div><div className="dataset-actions"><button className="secondary" onClick={()=>void downloadProcessedCsv()}><Download size={14}/> Download processed CSV</button><button className="secondary" onClick={()=>S('overview')}>View overview</button></div></div>}
 {tab==='home'&&!initialLoading&&<DashboardHome activeDataset={p?{dataset_id:p.dataset_id,filename:p.filename,rows:p.rows??p.shape?.[0],columns:p.columns??p.shape?.[1],missing_total:p.missing_total}:null} datasets={ws.saved_datasets||[]} jobs={jobs} loading={initialLoading} onUpload={upload} onOpenDataset={restore} onOpenTab={(nextTab:string)=>S(nextTab)}/> }
 {p&&tab==='overview'&&<div className="stack"><div className="metrics">{metric('Shape',(p.shape||[p.rows,p.columns]).join(' × '))}{metric('Missing cells',p.missing_total)}{metric('Duplicate rows',p.duplicate_rows)}{metric('Memory MB',p.memory_mb)}</div>{card('Dataset schema',table(p.columns_info.map((c:any)=>({column:c.column,type:c.dtype,missing:c.missing,missing_pct:c.missing_pct+'%',unique:c.unique,sample:(c.sample||[]).join(', ')}))))}{card('Detailed numeric statistics',table(Object.entries(p.column_stats||{}).map(([column,s]:any)=>({column,...s}))))}{card('Data preview',table(p.preview))}{card('Saved local workspace',<><div className="metrics">{metric('Datasets',ws.saved_datasets?.length||0)}{metric('Models',ws.saved_models||0)}{metric('Training jobs',jobs.length)}</div>{(ws.saved_datasets||[]).map((d:any)=><div className="savedrow" key={d.dataset_id}><span>{d.filename}<small>{d.saved_at}</small></span><button className="secondary" onClick={()=>restore(d.dataset_id)}>Restore</button></div>)}{jobs.filter((j:any)=>j.model_id&&j.model_available).slice(0,8).map((j:any)=><div className="savedrow" key={j.model_id}><span>{j.best_model} · {j.task} · {j.target}<small>{j.created_at} · {(j.model_size_bytes/1048576).toFixed(2)} MB</small></span><div className="model-download-actions"><button className="secondary" onClick={()=>download('/api/models/'+j.model_id+'/download')}>.joblib</button><button className="secondary" onClick={()=>download('/api/models/'+j.model_id+'/download-pkl')}>.pkl</button></div></div>)}</>)}</div>}
 {tab==='experiments'&&<div className="stack">{card('Experiment history',<><p>Experiments are stored in your workspace. If the server restarted during training, the saved configuration can be retried from the beginning.</p>{trainingQueue.length===0?<p>No experiments yet. Start a training run to build your history.</p>:trainingQueue.map((j:any)=><div className="experiment-row" key={j.job_id}><div className="experiment-info"><strong>{j.best_model||j.model_name||'AutoML experiment'}</strong><span>{j.task||'Task'} · Target: {j.target||'—'} · {j.status}</span><small>{j.created_at||'Date unavailable'}{j.retry_of?' · Retry of '+j.retry_of.slice(0,8):''}</small>{j.error&&<small className="experiment-error">{j.error}</small>}</div><div className="model-download-actions">{j.model_available&&<><button className="secondary" onClick={()=>download('/api/models/'+j.model_id+'/download')}>.joblib</button><button className="secondary" onClick={()=>download('/api/models/'+j.model_id+'/download-pkl')}>.pkl</button></>}{j.leaderboard?.length>0&&<button className="secondary" onClick={()=>downloadEvaluationResults(j)}>Results CSV</button>}{(j.status==='queued'||j.status==='running')&&<button className="stop-button compact-stop" onClick={()=>void cancelTraining(j.job_id)}><span className="stop-square"/>Cancel</button>}{(j.status==='interrupted'||j.status==='failed'||j.status==='cancelled')&&j.can_retry&&<button className="secondary" onClick={()=>void retryTraining(j.job_id)}><RotateCcw size={14}/> Retry</button>}</div></div>)}</>)}</div>}
 {tab==='models'&&<div className="stack">{card('Saved model library',<><p>Download trained model artifacts and reuse them for predictions. Each experiment stores its best-performing model as a portable Joblib pipeline, including its fitted preprocessing steps.</p><div className="metrics">{metric('Saved models',jobs.filter((j:any)=>j.model_id&&j.model_available).length)}{metric('Unavailable artifacts',jobs.filter((j:any)=>j.model_id&&!j.model_available).length)}</div>{jobs.filter((j:any)=>j.model_id).length===0?<p>No saved models yet. Train a model to create your first downloadable artifact.</p>:jobs.filter((j:any)=>j.model_id).map((j:any)=><div className="model-library-item" key={j.model_id}><div className="model-library-info"><strong>{j.best_model||'Trained model'}</strong><span>{j.task||'Task'} · Target: {j.target||'—'}</span><small>{j.created_at||'Date unavailable'} · {j.model_available?(j.model_size_bytes/1048576).toFixed(2)+' MB':'Artifact unavailable on server'}</small></div><div className="model-download-actions"><button className="secondary" disabled={!j.model_available} onClick={()=>download('/api/models/'+j.model_id+'/download')}><Download size={15}/> .joblib</button><button className="secondary" disabled={!j.model_available} onClick={()=>download('/api/models/'+j.model_id+'/download-pkl')}><Download size={15}/> .pkl</button><button className="secondary" disabled={!j.leaderboard?.length} onClick={()=>downloadEvaluationResults(j)}><Download size={15}/> Results CSV</button></div></div>)}</>)}{card('Using a downloaded model',<><p>The downloaded .joblib or .pkl file contains the fitted scikit-learn pipeline. Load it in a compatible Python environment with Joblib and the same scikit-learn version used for training.</p><pre className="code-sample">{'import joblib\n\nmodel = joblib.load("your-model.joblib")\npredictions = model["pipeline"].predict(new_data[model["feature_cols"]])'}</pre><p className="hint">Only load model files from sources you trust. Joblib and PKL downloads contain the same Python pickle-serialized pipeline with different file extensions. Only load files from sources you trust; pickle-based formats can execute code when loaded.</p></>)}</div>} {p&&tab==='prepare'&&<div className="stack">{card('Step 1 · Preprocessing operation',<><p>{op==='fill_missing'?(strategy==='drop'?'Drop rows containing missing values in selected columns, or all columns when none are selected.':(['mean','median','zero'].includes(strategy)?`Fill selected numeric columns using ${strategy}. If none are selected, all numeric columns are used.`:`Fill missing values using ${strategy} in selected columns, or all columns when none are selected.`)):op==='drop_missing_rows'?'Drop rows with missing values in selected columns, or any column when none are selected.':op==='drop_columns'?'Permanently remove selected columns. Select at least one.':op==='drop_high_missing'?'Remove columns whose missing-value percentage exceeds the selected threshold.':op==='drop_duplicates'?'Remove duplicate rows while keeping the first occurrence.':op==='drop_outliers'?'Remove rows outside the 1.5×IQR range for selected numeric columns, or all numeric columns if none are selected.':op==='clip_outliers'?'Clip selected numeric columns to their 1.5×IQR bounds.':op==='convert_numeric'?'Convert selected columns to numeric values; invalid values become missing.':op==='convert_datetime'?'Convert selected columns to datetime values.':op==='rename_column'?'Select exactly one column and enter a unique new name.':op==='sample_rows'?'Keep a reproducible random sample of the requested number of rows.':'Reset the row index.'} The dataset profile and missing-value counts refresh after a successful operation.</p><div className="formgrid">{field('Operation',<select value={op} onChange={e=>{SO(e.target.value);ER('');SC([])}}>{[['fill_missing','Handle null values'],['drop_duplicates','Remove duplicates'],['drop_missing_rows','Drop rows with nulls'],['drop_columns','Drop selected columns'],['drop_high_missing','Drop high-null columns'],['drop_outliers','Remove IQR outliers'],['clip_outliers','Clip IQR outliers'],['convert_numeric','Convert numeric'],['convert_datetime','Convert datetime'],['rename_column','Rename column'],['sample_rows','Sample rows'],['reset_index','Reset index']].map(([v,l])=><option key={v} value={v}>{l}</option>)}</select>)}{op==='fill_missing'&&field('Null strategy',<select value={strategy} onChange={e=>{SS(e.target.value);ER('');SC([])}}>{['median','mean','mode','constant','zero','ffill','bfill','drop'].map(v=><option key={v}>{v}</option>)}</select>)}{op==='fill_missing'&&strategy==='constant'&&field('Replacement value',<input value={value} onChange={e=>SV(e.target.value)}/>)}{op==='drop_high_missing'&&field('Missing threshold '+Math.round(threshold*100)+'%',<input type="range" min=".1" max="1" step=".05" value={threshold} onChange={e=>ST(+e.target.value)}/>)}{op==='sample_rows'&&field('Rows to keep',<input type="number" min="1" max={p.rows} value={sampleSize} onChange={e=>SSample(Math.max(1,Math.min(p.rows,Number(e.target.value)||1)))}/>)}{op==='rename_column'&&field('New column name',<input value={newName} onChange={e=>SN(e.target.value)}/>)}</div>{needsColumns&&<>{((op==='drop_outliers'||op==='clip_outliers'||(op==='fill_missing'&&['mean','median','zero'].includes(strategy)))&&prepColumns.length===0)&&<p className="hint">This operation requires numeric columns, but the dataset has none available.</p>}<h3>Columns {op==='rename_column'?'(select exactly one)':''}</h3><div className="checkgrid">{prepColumns.map((c:any)=><label key={c.column}><input type="checkbox" checked={cols.includes(c.column)} onChange={e=>SC(a=>e.target.checked?(op==='rename_column'?[c.column]:[...a,c.column]):a.filter(v=>v!==c.column))}/>{c.column} <small>({c.missing} null)</small></label>)}</div></>}<button className="primary" onClick={prep} disabled={busy||hasActiveTraining||!preprocessCanRun}>Apply and save</button></>)}{card('Step 2 · Feature engineering',<><p>Select the exact columns to transform. This selection is independent of preprocessing.</p><h3>Feature columns</h3><div className="checkgrid">{(p.columns_info||[]).map((c:any)=><label key={c.column}><input type="checkbox" checked={featureCols.includes(c.column)} onChange={e=>SFeatureCols(current=>e.target.checked?[...current,c.column]:current.filter(name=>name!==c.column))}/>{c.column} <small>({c.kind})</small></label>)}</div><div className="actions"><button onClick={()=>feat('one_hot_encode')} disabled={busy||hasActiveTraining||featureCols.length===0}>One-hot encode selected</button><button onClick={()=>feat('standardize')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Standardize selected</button><button onClick={()=>feat('minmax')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Min-max scale selected</button><button onClick={()=>feat('log_transform')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Log transform selected</button><button onClick={()=>feat('polynomial')} disabled={busy||hasActiveTraining||selectedNumericCount===0}>Squared features selected</button>{field('PCA components',<input type="number" min="1" max="20" value={components} onChange={e=>SCM(Math.max(1,Math.min(20,Number(e.target.value)||1)))}/>)}<button onClick={()=>feat('pca')} disabled={busy||hasActiveTraining||selectedNumericCount<2}>Run PCA on selected</button></div><small>{featureCols.length} columns selected · {selectedNumericCount} numeric</small></>)}{card('Dataset version history',<><p>Each upload, preprocessing operation and feature-engineering operation creates a recoverable snapshot. Restoring a version makes it the active dataset and records that restoration as a new version.</p>{datasetVersions.length===0?<p>No saved versions are available yet.</p>:datasetVersions.map((version:any)=><div className="savedrow" key={version.version_id}><span><strong>{version.reason}</strong><small>{version.created_at} · {version.rows} rows · {version.columns} columns · {(version.size_bytes/1048576).toFixed(2)} MB</small></span><button className="secondary" disabled={busy||hasActiveTraining} onClick={()=>restoreVersion(version.version_id)}><RotateCcw size={14}/> Restore</button></div>)}</>)}</div>}
 {p&&tab==='charts'&&<><style>{`@media print {
 @page { size: A4 landscape; margin: 10mm; }
 body * { visibility: hidden !important; }
 .chart-dashboard, .chart-dashboard * { visibility: visible !important; }
 .chart-dashboard { position: absolute !important; left: 0 !important; top: 0 !important; width: 100% !important; display: grid !important; gap: 12px !important; padding: 0 !important; background: #ffffff !important; }
 .chart-card { break-inside: avoid; page-break-inside: avoid; box-shadow: none !important; border: 1px solid #d8dee8 !important; color: #172033 !important; }
 [data-export-ignore="true"] { display: none !important; }
}`}</style><div className="stack">
{card('Chart dashboard',<>
<p>Configure a graph, select its columns and add it to the dashboard. Export individual graphs or print the complete layout.</p>
<div className="formgrid">
{field('Graph type',<select value={kind} onChange={e=>SK(e.target.value)}>{['bar','line','area','scatter','pie','count','histogram','heatmap','density','box','radar','ecdf','qq','step','pareto'].map(v=><option key={v} value={v}>{v.replace(/\b\w/g, (letter:string) => letter.toUpperCase())}</option>)}</select>)}
{kind!=='heatmap'&&field(kind==='histogram'?'Numeric column':kind==='scatter'?'Numeric X axis':'X axis / category',<select value={x} onChange={e=>SX(e.target.value)}><option value="">Select a column</option>{chartXColumns.map((col:any)=><option key={col.column} value={col.column}>{col.column}</option>)}</select>)}
{['line','area','scatter','bar','box','radar'].includes(kind)&&field('Numeric Y axis / value',<select value={y} onChange={e=>SY(e.target.value)}><option value="">Select a numeric column</option>{chartYColumns.map((col:any)=><option key={col.column} value={col.column}>{col.column}</option>)}</select>)}
{['bar','line','area','step','radar'].includes(kind)&&field('Aggregation',<select value={agg} onChange={e=>SA(e.target.value)}>{['mean','sum','median','count','min','max'].map(v=><option key={v}>{v}</option>)}</select>)}
{field('Max groups '+rows,<input type="range" min="5" max="100" step="5" value={rows} onChange={e=>SR(+e.target.value)}/>)}
{field('Dashboard columns',<select value={grid} onChange={e=>SG(+e.target.value)}><option value={1}>1 column</option><option value={2}>2 columns</option><option value={3}>3 columns</option></select>)}
</div>
<div className="actions"><button className="primary" onClick={makeChart} disabled={busy||(kind!=='heatmap'&&!x)||(['line','area','step','scatter'].includes(kind)&&!y)}>Add graph to dashboard</button>{Array.isArray(chart)&&chart.length>0&&<><button className="secondary" onClick={downloadDashboardPdf} disabled={busy}><Download size={14}/> Print / Save dashboard PDF</button><button className="secondary" onClick={()=>SCH([])}>Clear dashboard</button></>}</div>
</>)}
{Array.isArray(chart)&&chart.length>0&&<div className="chart-dashboard" style={{gridTemplateColumns:'repeat('+grid+',minmax(0,1fr))'}}>
{chart.map((item:any,index:number)=><ChartCard key={index} item={item} index={index} busy={busy} onDownload={downloadChartPng} onRemove={(removeIndex:number)=>SCH((current:any[])=>current.filter((_:any,i:number)=>i!==removeIndex))}/>)}
</div>}
</div></>}
{p&&tab==='train'&&<div className="stack">{card('Step 1 · Configure model',<><div className="formgrid">{field('Target',<select value={target} onChange={e=>{STG(e.target.value);SF(f=>f.filter(c=>c!==e.target.value))}}><option value="">Choose target</option>{targetColumns.map((c:any)=><option key={c.column} value={c.column}>{c.column} ({c.dtype})</option>)}</select>)}{field('Task',<select value={task} onChange={e=>{const next=e.target.value;STK(next);SNModel(next==='classification'?'Logistic Regression':'Ridge Regression');const eligible=(p.columns_info||[]).filter((c:any)=>c.unique>=2&&(next==='regression'?c.kind==='numeric':c.unique<=100));if(!eligible.some((c:any)=>c.column===target))STG(eligible[0]?.column||'')}}><option value="classification">Classification</option><option value="regression">Regression</option></select>)}{field('Model selection',<select value={mode} onChange={e=>SM(e.target.value)}><option value="auto">Select all algorithms</option><option value="manual">Choose multiple algorithms</option></select>)}{mode==='manual'&&<div className="model-picker"><label className="select-all-models"><input type="checkbox" checked={selectedModels.length===models.length} onChange={e=>SSelectedModels(e.target.checked?[...models]:[])}/>Select all models ({models.length})</label><div className="checkgrid">{models.map(m=><label key={m}><input type="checkbox" checked={selectedModels.includes(m)} onChange={e=>SSelectedModels(current=>e.target.checked?[...current,m]:current.filter(name=>name!==m))}/>{m}</label>)}</div><small>{selectedModels.length} of {models.length} models selected</small></div>}{field('Test split '+Math.round(split*100)+'%',<input type="range" min=".1" max=".5" step=".05" value={split} onChange={e=>SSplit(+e.target.value)}/>)}{field('Feature selection',<select value={selection} onChange={e=>SSEL(e.target.value)}><option value="none">None</option><option value="kbest">Select K best</option><option value="variance">Remove constant features</option></select>)}{selection==='kbest'&&field('K features '+k,<input type="range" min="1" max="50" value={k} onChange={e=>SKN(+e.target.value)}/>)}{field('PCA '+pcaK+' components',<><input type="checkbox" checked={pca} onChange={e=>SPCA(e.target.checked)}/><input type="range" min="1" max="20" value={pcaK} onChange={e=>SPK(+e.target.value)} disabled={!pca}/></>)}</div><label><input type="checkbox" checked={scale} onChange={e=>SSC(e.target.checked)}/> Scale numeric features</label> <label><input type="checkbox" checked={drop} onChange={e=>SD(e.target.checked)}/> Drop columns with more than 80% missing</label><h3>Input features</h3><p>{task==='classification'?'Classification predicts a category or class label.':'Regression predicts a continuous numeric value.'} Select at least one input feature. The target is excluded automatically.</p><div className="checkgrid">{p.columns_info.filter((c:any)=>c.column!==target).map((c:any)=><label key={c.column}><input type="checkbox" checked={features.includes(c.column)} onChange={e=>SF(a=>e.target.checked?[...a,c.column]:a.filter(v=>v!==c.column))}/>{c.column}</label>)}</div><p>Available {task} algorithms: {models.join(', ')}.</p><button className={hasActiveTraining?'stop-button':'primary'} onClick={()=>activeTrainingJob?void cancelTraining(activeTrainingJob.job_id):void train()} disabled={busy||Boolean(activeTrainingJob?.cancel_requested)||(!hasActiveTraining&&(!target||!features.some(c=>c!==target)||(mode==='manual'&&selectedModels.length===0)))}>{hasActiveTraining?(activeTrainingJob?.cancel_requested?<>Stopping…</>:<><span className="stop-square"/>Stop training</>) : <> <FlaskConical size={15}/>Train {mode==='auto'?'and compare all '+task+' models':selectedModels.length+' selected models'}</>}</button></>)}{trainingQueue.length>0&&card('Training queue',<div className="stack">{trainingQueue.filter((j:any)=>['queued','running'].includes(j.status)).map((j:any)=><div className="training-job-card" key={j.job_id}><div className="training-job-top"><strong>{j.model_name} · {j.task} · {j.target}</strong><span className="queue-status">{j.status}</span></div><div className="training-progress-track"><span style={{width:Math.max(0,Math.min(100,j.progress_pct||0))+'%'}}/></div><div className="training-progress-meta"><span>{j.current_model?'Current algorithm: '+j.current_model:(j.status==='queued'?'Waiting for available training worker':'Preparing models')} · {j.completed_models||0}/{j.total_models||'—'} completed</span><span>{j.progress_pct||0}%</span></div><small>{j.status==='queued'?'Queue position '+(j.queue_position||'—'):durationLabel(j.eta_seconds)}{j.elapsed_seconds!==undefined?' · Elapsed '+j.elapsed_seconds+'s':''}</small>{j.skip_requested_model&&<small>Skipping {j.skip_requested_model} after its current fit finishes.</small>}<div className="training-job-actions">{j.current_model&&!j.skip_requested_model&&<button className="secondary compact-stop" onClick={()=>void skipCurrentModel(j.job_id)}><RotateCcw size={13}/> Skip current algorithm</button>}{j.cancel_requested?<small>Cancellation requested; the current model fit will finish safely.</small>:<button className="stop-button compact-stop" onClick={()=>void cancelTraining(j.job_id)}><span className="stop-square"/>Cancel experiment</button>}</div></div>)}</div>)}</div>}
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
            <div className="actions"><button className="secondary" onClick={() => download('/api/models/' + model.model_id + '/download')}><Download size={16}/> .joblib</button><button className="secondary" onClick={() => download('/api/models/' + model.model_id + '/download-pkl')}><Download size={16}/> .pkl</button><button className="secondary" onClick={() => downloadEvaluationResults(model)}><Download size={16}/> Evaluation CSV</button></div>
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
 {p&&workflowNav.some((item:any)=>item[0]===tab)&&<div className="phase-nav"><div className="phase-heading"><div><small>ANALYSIS WORKFLOW</small><strong>Step {phaseIndex+1} of {workflowNav.length}: {workflowNav[phaseIndex]?.[1]}</strong></div><span>{Math.round(((phaseIndex+1)/workflowNav.length)*100)}% of workflow</span></div><div className="phase-progress" aria-label="Workflow progress">{workflowNav.map((item:any,i:number)=><button type="button" key={item[0]} aria-label={'Go to '+item[1]} aria-current={i===phaseIndex?'step':undefined} className={i===phaseIndex?'current':i<phaseIndex?'complete':''} onClick={()=>goToPhase(i)} title={item[1]}/>)}</div><div className="phase-controls"><button className="phase-back" onClick={()=>goToPhase(Math.max(0,phaseIndex-1))} disabled={phaseIndex===0}><ChevronLeft size={15}/>Previous</button><button className="primary phase-next" onClick={()=>goToPhase(phaseIndex===workflowNav.length-1?0:phaseIndex+1)}>{phaseIndex===workflowNav.length-1?'Return to overview':'Continue'}<ChevronRight size={15}/></button></div></div>}
 <footer>AutoML Studio · React + FastAPI · Local-first</footer></section></main></div>
}
function App(){
 const [user,setUser]=useState<{id:string;email:string}|null>(null)
 const [ready,setReady]=useState(false)
 const [path,setPath]=useState(window.location.pathname)
 function navigate(to:string,replace=false){
  if(replace)window.history.replaceState({},'',to)
  else if(window.location.pathname+window.location.search!==to)window.history.pushState({},'',to)
  setPath(window.location.pathname)
  window.scrollTo({top:0,behavior:'auto'})
 }
 useEffect(()=>{const pop=()=>setPath(window.location.pathname);window.addEventListener('popstate',pop);return()=>window.removeEventListener('popstate',pop)},[])
 useEffect(()=>{(async()=>{try{let result=await authClient.getSession();if(!result.error&&result.data?.user&&result.data.session?.token){setUser(result.data.user);return}result=await authClient.getSession({fetchOptions:{headers:{'X-Force-Fetch':'true'}}});if(result.error)throw new Error(result.error.message);if(result.data?.user&&result.data.session?.token)setUser(result.data.user)}catch(error){console.error('Could not restore the sign-in session:',error)}finally{setReady(true)}})()},[])
 useEffect(()=>{
  if(!ready)return
  if(user&&['/','/login','/signup'].includes(path))navigate('/app/home',true)
  else if(user&&path==='/app')navigate('/app/home',true)
  else if(!user&&(path==='/app'||path.startsWith('/app/')))navigate('/login?next='+encodeURIComponent(path==='/app'?'/app/home':path),true)
 },[ready,user,path])
 async function signOut(){try{await authClient.signOut()}catch{}setUser(null);navigate('/login',true)}
 if(!ready)return <main className="route-loading"><div className="route-loading-mark">A</div><p>Restoring your workspace session</p><span>Checking account access…</span></main>
 const cleanPath=path.endsWith('/')&&path!=='/'?path.slice(0,-1):path
 const statusSegment=cleanPath.split('/').filter(Boolean).pop()||''
 const statusCode=Number(statusSegment)
 const isStatusPage=['400','401','403','404','405','408','409','410','413','422','429','500','501','502','503','504'].includes(statusSegment)&&(cleanPath==='/'+statusSegment||cleanPath==='/status/'+statusSegment)
 if(isStatusPage)return <SystemPage code={statusCode} onHome={()=>navigate(user?'/app/home':'/')} onLogin={()=>navigate('/login')}/>
 
 if(cleanPath==='/privacy'||cleanPath==='/terms'||cleanPath==='/about'||cleanPath==='/contact'||cleanPath==='/'){
  const publicRoute=cleanPath==='/'?'home':cleanPath.slice(1) as 'about'|'privacy'|'terms'|'contact'
  return <PublicPage page={publicRoute} onLogin={()=>navigate('/login')} onSignup={()=>navigate('/signup')}/>
 }
 if(cleanPath==='/login'||cleanPath==='/signup'){
  if(user)return null
  return <AuthScreen initialMode={cleanPath==='/signup'?'signup':'login'} onModeChange={mode=>navigate((mode==='signup'?'/signup':'/login')+window.location.search,true)} onAuthenticated={account=>{
   setUser(account)
   const next=new URLSearchParams(window.location.search).get('next')
   navigate(next&&next.startsWith('/app/')?next:'/app/home',true)
  }}/>
 }
 if(user&&cleanPath.startsWith('/app/')){
  const tab=cleanPath.slice('/app/'.length).split('/')[0]
  const validTabs=['home','overview','prepare','charts','train','evaluation','predict','models','experiments']
  if(validTabs.includes(tab))return <Workspace key={user.id} user={user} onSignOut={signOut} initialTab={tab} onNavigate={next=>navigate('/app/'+next)}/>
 }
 if(cleanPath.startsWith('/app/'))return <SystemPage code={404} onHome={()=>navigate(user?'/app/home':'/')} onLogin={()=>navigate('/login')}/>
 return <SystemPage code={404} onHome={()=>navigate(user?'/app/home':'/')} onLogin={()=>navigate('/login')}/>
}
export default App
