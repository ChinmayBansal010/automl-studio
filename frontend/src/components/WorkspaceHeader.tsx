import type { ChangeEventHandler, RefObject } from 'react'
import { FilePlus2 } from 'lucide-react'

type Props = {
  title: string
  datasetName?: string
  disabled: boolean
  uploadInputRef: RefObject<HTMLInputElement>
  onFileChange: ChangeEventHandler<HTMLInputElement>
}

export function WorkspaceHeader({ title, datasetName, disabled, uploadInputRef, onFileChange }: Props) {
  return (
    <header className="workspace-header">
      <div className="header-title">
        <small>MACHINE LEARNING WORKSPACE</small>
        <h1>{title}</h1>
        {datasetName && <span className="filetag header-filetag" title={datasetName}>{datasetName}</span>}
      </div>
      <div className="header-actions">
        <button className="new-task-button" type="button" onClick={() => uploadInputRef.current?.click()} disabled={disabled}>
          <FilePlus2 size={16}/>New task
        </button>
        <input ref={uploadInputRef} className="hidden-upload" type="file" accept=".csv"
          onChange={onFileChange}/>
      </div>
    </header>
  )
}
