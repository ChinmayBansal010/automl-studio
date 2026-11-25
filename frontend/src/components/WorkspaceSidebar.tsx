import { LogOut, Sparkles } from 'lucide-react'

type Props = {
  items: any[]
  activeTab: string
  email: string
  onNavigate: (tab: string) => void
  onSignOut: () => void | Promise<void>
}

export function WorkspaceSidebar({ items, activeTab, email, onNavigate, onSignOut }: Props) {
  const renderItems = (entries: any[]) => entries.map(([id, label, Icon]) => (
    <button key={id} type="button" aria-current={activeTab === id ? 'page' : undefined}
      className={activeTab === id ? 'active' : ''} onClick={() => onNavigate(id)}>
      <Icon size={17}/><span>{label}</span>
    </button>
  ))

  return (
    <aside className="sidebar">
      <div className="brand"><Sparkles/><div><b>AutoML Studio</b><small>PRIVATE DATA WORKSPACE</small></div></div>
      <nav aria-label="Workspace navigation">
        <div className="nav-section-label">HOME</div>
        {renderItems(items.slice(0, 1))}
        <div className="nav-section-label nav-section-secondary">ANALYSIS</div>
        {renderItems(items.slice(1, 7))}
        <div className="nav-section-label nav-section-secondary">LIBRARY</div>
        {renderItems(items.slice(7))}
      </nav>
      <div className="sidebottom">
        <span className="signed-in">{email}</span>
        <button className="signout" type="button" onClick={() => void onSignOut()}><LogOut size={14}/>Sign out</button>
      </div>
    </aside>
  )
}
