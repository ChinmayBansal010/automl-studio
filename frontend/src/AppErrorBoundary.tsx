import React from 'react'
import { SystemPage } from './SystemPage'

type Props = { children: React.ReactNode }
type State = { failed: boolean }

export class AppErrorBoundary extends React.Component<Props, State> {
  state: State = { failed: false }
  static getDerivedStateFromError(): State { return { failed: true } }
  componentDidCatch(error: Error) { console.error('Application render error:', error) }
  render() {
    if (this.state.failed) return <SystemPage code={500} onHome={() => { window.location.href = '/' }} onLogin={() => { window.location.href = '/login' }}/>
    return this.props.children
  }
}
