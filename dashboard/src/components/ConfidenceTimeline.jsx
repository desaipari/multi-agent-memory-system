import { useState, useEffect, useCallback } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts"
import { memoryApi } from "../api"

const AGENT_COLORS = {
  intake_agent:      "#7c3aed",
  delivery_agent:    "#2563eb",
  billing_agent:     "#dc2626",
  coordinator_agent: "#059669"
}

export default function ConfidenceTimeline({ refreshTrigger }) {
  const [facts, setFacts] = useState([])
  const [selectedEntity, setSelectedEntity] = useState("")
  const [selectedFactType, setSelectedFactType] = useState("")
  const [loading, setLoading] = useState(true)

  const fetchFacts = useCallback(async () => {
    try {
      const data = await memoryApi.getAllFacts()
      setFacts(data.facts || [])
    } catch (err) {
      console.error("Cannot fetch facts:", err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchFacts()
  }, [fetchFacts, refreshTrigger])

  useEffect(() => {
    const interval = setInterval(fetchFacts, 5000)
    return () => clearInterval(interval)
  }, [fetchFacts])

  // Get unique entities and fact types for filter dropdowns
  const entities = [...new Set(facts.map(f => f.entity))].sort()
  const factTypes = [...new Set(facts.map(f => f.fact_type))].sort()

  // Set defaults when data loads
  useEffect(() => {
    if (entities.length > 0 && !selectedEntity) {
      setSelectedEntity(entities[0])
    }
    if (factTypes.length > 0 && !selectedFactType) {
      setSelectedFactType(factTypes[0])
    }
  }, [entities, factTypes])

  // Build timeline data for selected entity + fact type
  const filteredFacts = facts
    .filter(f =>
      f.entity === selectedEntity &&
      f.fact_type === selectedFactType
    )
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))

  const chartData = filteredFacts.map((fact, index) => ({
    index: index + 1,
    confidence: Math.round(fact.confidence * 100),
    agent: fact.agent_id,
    value: fact.value,
    status: fact.status,
    time: new Date(fact.timestamp).toLocaleTimeString()
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null
    const d = payload[0].payload
    return (
      <div style={{
        backgroundColor: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "6px",
        padding: "10px 14px",
        fontSize: "12px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
      }}>
        <div style={{ fontWeight: "600", marginBottom: "4px" }}>
          Write #{d.index}
        </div>
        <div>Agent: <span style={{ color: AGENT_COLORS[d.agent] || "#6b7280" }}>{d.agent}</span></div>
        <div>Value: {d.value}</div>
        <div>Confidence: <strong>{d.confidence}%</strong></div>
        <div>Status: {d.status}</div>
        <div>Time: {d.time}</div>
      </div>
    )
  }

  if (loading) return (
    <div style={{
      backgroundColor: "white", borderRadius: "8px",
      padding: "40px", textAlign: "center"
    }}>
      Loading timeline data...
    </div>
  )

  return (
    <div style={{
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
      overflow: "hidden"
    }}>
      <div style={{
        padding: "14px 20px",
        borderBottom: "1px solid #e5e7eb"
      }}>
        <h3 style={{ margin: "0 0 10px 0", fontSize: "15px" }}>
          Confidence Timeline
        </h3>
        <p style={{ margin: "0 0 10px 0", fontSize: "12px", color: "#9ca3af" }}>
          How confidence in a specific fact changed over time
          as agents wrote and corroborated it
        </p>
        <div style={{ display: "flex", gap: "10px" }}>
          <select
            value={selectedEntity}
            onChange={e => setSelectedEntity(e.target.value)}
            style={{
              padding: "5px 8px",
              borderRadius: "4px",
              border: "1px solid #d1d5db",
              fontSize: "12px"
            }}
          >
            {entities.map(e => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>
          <select
            value={selectedFactType}
            onChange={e => setSelectedFactType(e.target.value)}
            style={{
              padding: "5px 8px",
              borderRadius: "4px",
              border: "1px solid #d1d5db",
              fontSize: "12px"
            }}
          >
            {factTypes.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ padding: "20px" }}>
        {chartData.length === 0 ? (
          <div style={{
            textAlign: "center",
            color: "#9ca3af",
            fontSize: "13px",
            padding: "40px"
          }}>
            No data for {selectedEntity} / {selectedFactType}
            <br />
            Write some facts first using the input panel
          </div>
        ) : (
          <LineChart
            width={600}
            height={300}
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="index"
              label={{
                value: "Write sequence",
                position: "insideBottom",
                offset: -5,
                fontSize: 11
              }}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
              label={{
                value: "Confidence",
                angle: -90,
                position: "insideLeft",
                fontSize: 11
              }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Auto-resolve threshold line */}
            <ReferenceLine
              y={70}
              stroke="#22c55e"
              strokeDasharray="5 5"
              label={{
                value: "Action threshold (70%)",
                position: "right",
                fontSize: 10,
                fill: "#22c55e"
              }}
            />

            {/* Auto-resolve threshold */}
            <ReferenceLine
              y={30}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{
                value: "Min confidence (30%)",
                position: "right",
                fontSize: 10,
                fill: "#f59e0b"
              }}
            />

            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={(props) => {
                const { cx, cy, payload } = props
                const color = AGENT_COLORS[payload.agent] || "#6b7280"
                const isContested = payload.status === "contested"
                const isSuperseded = payload.status === "superseded"
                return (
                  <circle
                    key={`dot-${props.index}`}
                    cx={cx}
                    cy={cy}
                    r={5}
                    fill={isContested ? "#f59e0b" : isSuperseded ? "#9ca3af" : color}
                    stroke="white"
                    strokeWidth={2}
                  />
                )
              }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        )}

        {/* Legend */}
        {chartData.length > 0 && (
          <div style={{
            marginTop: "16px",
            display: "flex",
            gap: "16px",
            flexWrap: "wrap",
            fontSize: "11px"
          }}>
            {Object.entries(AGENT_COLORS).map(([agent, color]) => (
              <div key={agent} style={{
                display: "flex",
                alignItems: "center",
                gap: "4px"
              }}>
                <div style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  backgroundColor: color
                }} />
                <span style={{ color: "#6b7280" }}>{agent}</span>
              </div>
            ))}
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{
                width: "10px", height: "10px",
                borderRadius: "50%", backgroundColor: "#f59e0b"
              }} />
              <span style={{ color: "#6b7280" }}>contested</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}