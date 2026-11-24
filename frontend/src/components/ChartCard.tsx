import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar,
  Legend, Line, LineChart, Pie, PieChart, PolarAngleAxis, PolarGrid,
  PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis
} from 'recharts'
import { Download, X } from 'lucide-react'

type Props = {
  item: any
  index: number
  busy: boolean
  onDownload: (index: number) => void
  onRemove: (index: number) => void
}

function renderTableContent(data: any[]) {
  if (!data?.length) return <p>No rows to display.</p>
  const columns = Object.keys(data[0])
  return <div className="tablewrap"><table><thead><tr>{columns.map(column => <th key={column}>{column}</th>)}</tr></thead><tbody>{data.slice(0, 100).map((row, rowIndex) => <tr key={rowIndex}>{columns.map(column => <td key={column}>{row[column] === null || row[column] === undefined ? '—' : String(row[column])}</td>)}</tr>)}</tbody></table></div>
}

function renderTable(data: any[]) {
  if (!data?.length) return <p>No rows to display.</p>
  return data.length > 12
    ? <details className="table-disclosure"><summary>View chart data · {data.length.toLocaleString()} rows (first 100 displayed)</summary>{renderTableContent(data)}</details>
    : renderTableContent(data)
}

export function ChartCard({ item, index, busy, onDownload, onRemove }: Props) {
  const title = String(item.chart_type || 'graph').replace(/\b\w/g, (letter: string) => letter.toUpperCase())
  const data = Array.isArray(item.data) ? item.data : []
  return (
    <section className="card chart-card">
      <div className="chart-card-heading">
        <div className="chart-card-title"><span>GRAPH {String(index + 1).padStart(2, '0')}</span><h2>{title}</h2></div>
        <div className="actions" data-export-ignore="true">
          <button className="secondary" type="button" onClick={() => onDownload(index)} disabled={busy}><Download size={13}/> PNG</button>
          <button className="secondary chart-remove-button" type="button" onClick={() => onRemove(index)} aria-label={'Remove ' + title + ' chart'} title="Remove graph"><X size={14}/></button>
        </div>
      </div>
      {item.chart_type === 'heatmap' ? (data.length ? renderTableContent(data) : <p>{item.message || 'No numeric columns are available for this heatmap.'}</p>) : (
        <ResponsiveContainer width="100%" height={320}>
          {item.chart_type === 'pie' ? <PieChart><Tooltip/><Legend/><Pie data={data} dataKey="value" nameKey="x" outerRadius={105} label>{data.map((_: any, i: number) => <Cell key={i} fill={['#315fba','#4b8b75','#c18a45','#8a75b6','#b66c72'][i % 5]}/>)}</Pie></PieChart>
          : item.chart_type === 'scatter' ? <ScatterChart><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" type="number" label={{value:item.x_label || 'X',position:'insideBottom',offset:-5}}/><YAxis dataKey="value" label={{value:item.y_label || 'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Scatter data={data} fill="#315fba"/></ScatterChart>
          : item.chart_type === 'radar' ? <RadarChart data={data}><PolarGrid/><PolarAngleAxis dataKey="x"/><PolarRadiusAxis/><Tooltip/><Radar dataKey="value" stroke="#315fba" fill="#315fba" fillOpacity={.2}/></RadarChart>
          : item.chart_type === 'box' ? <ComposedChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label || 'Category',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label || 'Value',angle:-90,position:'insideLeft'}}/><Tooltip/><Bar dataKey="median" fill="#315fba"><ErrorBar dataKey="range" direction="y" width={5}/></Bar></ComposedChart>
          : item.chart_type === 'qq' ? <ScatterChart><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="theoretical" type="number" label={{value:'Theoretical quantiles',position:'insideBottom',offset:-5}}/><YAxis dataKey="observed" type="number" label={{value:'Observed quantiles',angle:-90,position:'insideLeft'}}/><Tooltip/><Scatter data={data} fill="#315fba"/></ScatterChart>
          : item.chart_type === 'pareto' ? <ComposedChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x"/><YAxis yAxisId="count"/><YAxis yAxisId="percent" orientation="right" domain={[0,100]}/><Tooltip/><Legend/><Bar yAxisId="count" dataKey="value" fill="#315fba"/><Line yAxisId="percent" dataKey="cumulative_pct" stroke="#c18a45"/></ComposedChart>
          : item.chart_type === 'step' ? <LineChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x"/><YAxis/><Tooltip/><Line dataKey="value" type="stepAfter" stroke="#315fba"/></LineChart>
          : item.chart_type === 'ecdf' ? <LineChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" type="number" label={{value:item.x_label || 'Value',position:'insideBottom',offset:-5}}/><YAxis domain={[0,1]} label={{value:'Cumulative probability',angle:-90,position:'insideLeft'}}/><Tooltip/><Line dataKey="value" stroke="#315fba" dot={false}/></LineChart>
          : item.chart_type === 'violin' ? <ComposedChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x"/><YAxis/><Tooltip/><Bar dataKey="median" fill="#315fba"><ErrorBar dataKey="range" direction="y" width={8}/></Bar></ComposedChart>
          : item.chart_type === 'density' ? <AreaChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" type="number" label={{value:item.x_label || 'Value',position:'insideBottom',offset:-5}}/><YAxis label={{value:'Density',angle:-90,position:'insideLeft'}}/><Tooltip/><Area dataKey="value" stroke="#4b8b75" fill="#4b8b75" fillOpacity={.18}/></AreaChart>
          : item.chart_type === 'line' ? <LineChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label || 'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label || 'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Line dataKey="value" stroke="#315fba"/></LineChart>
          : item.chart_type === 'area' ? <AreaChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label || 'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label || 'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Area dataKey="value" stroke="#315fba" fill="#315fba" fillOpacity={.16}/></AreaChart>
          : <BarChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="x" label={{value:item.x_label || 'X',position:'insideBottom',offset:-5}}/><YAxis label={{value:item.y_label || 'Y',angle:-90,position:'insideLeft'}}/><Tooltip/><Bar dataKey="value" fill="#315fba"/></BarChart>}
        </ResponsiveContainer>
      )}
      {item.chart_type !== 'heatmap' && renderTable(data)}
    </section>
  )
}
