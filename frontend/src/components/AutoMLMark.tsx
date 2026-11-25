import type { SVGProps } from 'react'
type Props = SVGProps<SVGSVGElement>
/** Original AutoML Studio identity: connected data points converging into a validated model node. */
export function AutoMLMark(props: Props) {
 return <svg viewBox="0 0 48 48" fill="none" role="img" aria-label="AutoML Studio" {...props}>
 <path d="M9 13.5 19.2 20M9 34.5 19.2 28M29 20l10-6.5M29 28l10 6.5" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" opacity=".82"/>
 <path d="M19.2 20 24 17.2 28.8 20v8L24 30.8 19.2 28v-8Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
 <path d="m21.5 24 1.8 1.8 3.4-3.8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
 <path d="M24 11.5v5.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" opacity=".75"/>
 <circle cx="7.5" cy="12.5" r="3.1" fill="currentColor"/><circle cx="7.5" cy="35.5" r="3.1" fill="currentColor"/><circle cx="40.5" cy="12.5" r="3.1" fill="currentColor"/><circle cx="40.5" cy="35.5" r="3.1" fill="currentColor"/><circle cx="24" cy="9" r="2.2" fill="currentColor" opacity=".75"/>
 </svg>
}