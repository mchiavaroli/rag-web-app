'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import useSWR from 'swr'
import { RefreshCw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { WikiPage } from '@/lib/types'

// ─────────────────────────────────────────────
// Tipi
// ─────────────────────────────────────────────
interface GraphNode {
  id: string
  label: string
  category: string
  links_count: number
  x: number
  y: number
}

interface GraphEdge {
  id: string
  source: string
  target: string
}

interface GraphData {
  nodes: Omit<GraphNode, 'x' | 'y'>[]
  edges: GraphEdge[]
}

// ─────────────────────────────────────────────
// Palette colori per categoria
// ─────────────────────────────────────────────
const CATEGORY_COLOR: Record<string, string> = {
  sources:    '#6366f1',
  concepts:   '#22c55e',
  procedures: '#f59e0b',
  components: '#3b82f6',
  images:     '#ec4899',
}

const CATEGORY_LABEL: Record<string, string> = {
  sources:    'Fonti',
  concepts:   'Concetti',
  procedures: 'Procedure',
  components: 'Componenti',
  images:     'Immagini',
}

// Raggio del nodo in base al numero di link
function nodeRadius(links_count: number) {
  return Math.max(18, Math.min(36, 18 + links_count * 2.5))
}

// ─────────────────────────────────────────────
// Layout radiale per categoria
// ─────────────────────────────────────────────
function computeLayout(nodes: GraphData['nodes']): GraphNode[] {
  // Raggruppa per categoria
  const grouped: Record<string, GraphData['nodes']> = {}
  for (const n of nodes) {
    if (!grouped[n.category]) grouped[n.category] = []
    grouped[n.category].push(n)
  }

  // Raggio e angolo di partenza per ogni categoria
  const RING: Record<string, { r: number; startAngle: number }> = {
    sources:    { r: 0,   startAngle: 0 },
    components: { r: 170, startAngle: Math.PI / 6 },
    procedures: { r: 170, startAngle: Math.PI * 5 / 6 },
    concepts:   { r: 320, startAngle: 0 },
    images:     { r: 490, startAngle: Math.PI / 8 },
  }

  const positioned: GraphNode[] = []

  for (const [cat, catNodes] of Object.entries(grouped)) {
    const ring = RING[cat] ?? { r: 380, startAngle: 0 }
    const count = catNodes.length

    if (ring.r === 0) {
      // Centro
      for (const n of catNodes) {
        positioned.push({ ...n, x: 0, y: 0 })
      }
    } else {
      // Distribuisci uniformemente sull'arco del ring
      // Le categorie piccole occupano un arco di 120° invece di 360°
      const spreadAngle = count <= 3 ? (Math.PI * 2) / 3 : Math.PI * 2
      for (let i = 0; i < count; i++) {
        const angle = ring.startAngle + (spreadAngle * i) / Math.max(1, count - (count <= 3 ? 0 : 0))
        positioned.push({
          ...catNodes[i],
          x: ring.r * Math.cos(angle),
          y: ring.r * Math.sin(angle),
        })
      }
    }
  }

  return positioned
}

// ─────────────────────────────────────────────
// Componente principale
// ─────────────────────────────────────────────
const fetcher = (url: string) => fetch(url).then(r => r.json())

interface WikiGraphProps {
  pages: WikiPage[]
  onSelectPage: (page: WikiPage) => void
}

export default function WikiGraph({ pages, onSelectPage }: WikiGraphProps) {
  const { data, isLoading, mutate } = useSWR<GraphData>('/api/wiki/graph', fetcher)

  // Trasformazione pan/zoom
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 })
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const dragStart = useRef<{ mx: number; my: number; tx: number; ty: number } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Calcola posizioni nodi
  const graphNodes: GraphNode[] = data?.nodes ? computeLayout(data.nodes) : []
  const nodeMap = new Map(graphNodes.map(n => [n.id, n]))
  const edges: GraphEdge[] = data?.edges ?? []

  // Centro la vista al primo caricamento
  useEffect(() => {
    if (graphNodes.length > 0 && containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect()
      setTransform({ x: width / 2, y: height / 2, scale: 1 })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  // Pan con mouse
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    setIsDragging(true)
    dragStart.current = { mx: e.clientX, my: e.clientY, tx: transform.x, ty: transform.y }
  }, [transform])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragStart.current) return
    const { mx, my, tx, ty } = dragStart.current
    const dx = e.clientX - mx
    const dy = e.clientY - my
    setTransform(t => ({ ...t, x: tx + dx, y: ty + dy }))
  }, [isDragging])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    dragStart.current = null
  }, [])

  // Zoom con rotellina
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.1 : 0.91
    setTransform(t => {
      const newScale = Math.max(0.2, Math.min(4, t.scale * factor))
      // Zoom centrato sul puntatore
      const rect = containerRef.current?.getBoundingClientRect()
      if (!rect) return { ...t, scale: newScale }
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const newX = px - (px - t.x) * (newScale / t.scale)
      const newY = py - (py - t.y) * (newScale / t.scale)
      return { x: newX, y: newY, scale: newScale }
    })
  }, [])

  const zoomIn = () => setTransform(t => ({ ...t, scale: Math.min(4, t.scale * 1.2) }))
  const zoomOut = () => setTransform(t => ({ ...t, scale: Math.max(0.2, t.scale / 1.2) }))
  const resetView = useCallback(() => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect()
      setTransform({ x: width / 2, y: height / 2, scale: 1 })
    }
  }, [])

  // Click su nodo → apri pagina wiki
  const handleNodeClick = useCallback((nodeId: string) => {
    const page = pages.find(p => p.name === nodeId + '.md' || p.name === nodeId)
    if (page) onSelectPage(page)
  }, [pages, onSelectPage])

  // Genera path curvo tra due nodi
  function edgePath(src: GraphNode, dst: GraphNode): string {
    const mx = (src.x + dst.x) / 2
    const my = (src.y + dst.y) / 2
    // Curvatura leggera perpendicolare alla direzione
    const dx = dst.x - src.x
    const dy = dst.y - src.y
    const len = Math.sqrt(dx * dx + dy * dy) || 1
    const bend = Math.min(len * 0.2, 40)
    const cx = mx - (dy / len) * bend
    const cy = my + (dx / len) * bend
    return `M ${src.x} ${src.y} Q ${cx} ${cy} ${dst.x} ${dst.y}`
  }

  // Nodi e archi coinvolti nell'hover
  const hoveredNeighbors = new Set<string>()
  const hoveredEdges = new Set<string>()
  if (hoveredId) {
    for (const e of edges) {
      if (e.source === hoveredId || e.target === hoveredId) {
        hoveredNeighbors.add(e.source)
        hoveredNeighbors.add(e.target)
        hoveredEdges.add(e.id)
      }
    }
  }
  const isFiltered = hoveredId !== null

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden bg-background select-none"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
    >
      {/* Controlli */}
      <div className="absolute top-3 right-3 z-10 flex flex-col gap-1.5">
        <Button size="icon" variant="outline" className="h-7 w-7" onClick={zoomIn} title="Zoom in">
          <ZoomIn className="h-3.5 w-3.5" />
        </Button>
        <Button size="icon" variant="outline" className="h-7 w-7" onClick={zoomOut} title="Zoom out">
          <ZoomOut className="h-3.5 w-3.5" />
        </Button>
        <Button size="icon" variant="outline" className="h-7 w-7" onClick={resetView} title="Reset vista">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
        <Button size="icon" variant="outline" className="h-7 w-7" onClick={() => mutate()} title="Aggiorna">
          <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {/* Legenda */}
      <div className="absolute bottom-3 left-3 z-10 bg-card/90 backdrop-blur border border-border rounded-lg px-3 py-2 text-[11px] space-y-1">
        {Object.entries(CATEGORY_LABEL).map(([cat, label]) => (
          <div key={cat} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: CATEGORY_COLOR[cat] }}
            />
            <span className="text-muted-foreground">{label}</span>
          </div>
        ))}
        <p className="text-muted-foreground/60 pt-0.5 border-t border-border mt-1">
          Scroll per zoom · Trascina per pan
        </p>
      </div>

      {/* Stato caricamento / vuoto */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <RefreshCw className="h-8 w-8 text-muted-foreground animate-spin" />
        </div>
      )}
      {!isLoading && graphNodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
          Wiki vuota — esegui Ingest per popolare il grafo.
        </div>
      )}

      {/* SVG */}
      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{ touchAction: 'none' }}
      >
        <defs>
          {/* Freccia per ogni categoria */}
          {Object.entries(CATEGORY_COLOR).map(([cat, color]) => (
            <marker
              key={cat}
              id={`arrow-${cat}`}
              markerWidth="8"
              markerHeight="8"
              refX="6"
              refY="3"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L0,6 L8,3 z" fill={color} fillOpacity="0.7" />
            </marker>
          ))}
        </defs>

        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
          {/* ── Archi ── */}
          {edges.map(edge => {
            const src = nodeMap.get(edge.source)
            const dst = nodeMap.get(edge.target)
            if (!src || !dst) return null
            const isActive = !isFiltered || hoveredEdges.has(edge.id)
            const color = CATEGORY_COLOR[src.category] ?? '#888'
            return (
              <path
                key={edge.id}
                d={edgePath(src, dst)}
                stroke={color}
                strokeWidth={isActive ? 1.8 : 0.6}
                strokeOpacity={isActive ? 0.65 : 0.1}
                fill="none"
                markerEnd={isActive ? `url(#arrow-${src.category})` : undefined}
                style={{ transition: 'stroke-opacity 0.2s, stroke-width 0.2s' }}
              />
            )
          })}

          {/* ── Nodi ── */}
          {graphNodes.map(node => {
            const r = nodeRadius(node.links_count)
            const color = CATEGORY_COLOR[node.category] ?? '#888'
            const isHovered = hoveredId === node.id
            const isNeighbor = hoveredNeighbors.has(node.id)
            const dimmed = isFiltered && !isHovered && !isNeighbor
            const scale = isHovered ? 1.18 : 1

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                style={{
                  cursor: 'pointer',
                  opacity: dimmed ? 0.2 : 1,
                  transition: 'opacity 0.2s',
                }}
                onMouseEnter={() => setHoveredId(node.id)}
                onMouseLeave={() => setHoveredId(null)}
                onClick={(e) => { e.stopPropagation(); handleNodeClick(node.id) }}
              >
                {/* Alone su hover */}
                {isHovered && (
                  <circle
                    r={r * scale + 6}
                    fill={color}
                    fillOpacity={0.12}
                    style={{ transition: 'r 0.15s' }}
                  />
                )}
                {/* Cerchio principale */}
                <circle
                  r={r * scale}
                  fill={color}
                  fillOpacity={isHovered ? 1 : 0.82}
                  stroke="var(--background)"
                  strokeWidth={isHovered ? 2.5 : 1.5}
                  style={{ transition: 'r 0.15s, fill-opacity 0.15s' }}
                />
                {/* Etichetta */}
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={Math.max(9, Math.min(12, r * 0.55))}
                  fontWeight={isHovered ? 600 : 400}
                  fill="white"
                  style={{
                    pointerEvents: 'none',
                    userSelect: 'none',
                    textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                  }}
                >
                  {node.label.length > 14 ? node.label.slice(0, 13) + '…' : node.label}
                </text>
                {/* Label estesa sotto il nodo (solo su hover) */}
                {isHovered && (
                  <text
                    y={r * scale + 14}
                    textAnchor="middle"
                    dominantBaseline="hanging"
                    fontSize={11}
                    fontWeight={600}
                    fill={color}
                    stroke="var(--background)"
                    strokeWidth={3}
                    paintOrder="stroke"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >
                    {node.label}
                  </text>
                )}
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}
