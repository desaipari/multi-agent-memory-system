import { useCallback, useEffect, useState } from "react"
import { memoryApi } from "../api"

const AGENT_COLORS = {
  intake_agent: "#7c3aed",
  delivery_agent: "#2563eb",
  billing_agent: "#dc2626",
  billing_ops_agent: "#dc2626",
  coordinator_agent: "#059669",
}

function formatAgentName(agentId) {
  if (!agentId) return "Unknown Agent"

  const names = {
    intake_agent: "Intake Agent",
    delivery_agent: "Delivery Agent",
    billing_agent: "Billing Agent",
    billing_ops_agent: "Billing/Ops Agent",
    coordinator_agent: "Coordinator Agent",
  }

  return (
    names[agentId] ||
    agentId
      .split("_")
      .map(
        (word) =>
          word.charAt(0).toUpperCase() + word.slice(1)
      )
      .join(" ")
  )
}

export default function AgentTrustPanel({ refreshTrigger = 0 }) {
  const [agents, setAgents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchAgents = useCallback(async () => {
    try {
      setError(null)

      const data = await memoryApi.getAgents()
      setAgents(data.agents || [])
    } catch (err) {
      console.error("Cannot fetch agents:", err)
      setError("Unable to load agent trust data.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents, refreshTrigger])

  useEffect(() => {
    const interval = setInterval(fetchAgents, 5000)

    return () => clearInterval(interval)
  }, [fetchAgents])

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Agent Trust Scoreboard</h2>
            <p>Loading agent data...</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Agent Trust Scoreboard</h2>

          <p>
            Trust scores update after agent writes and
            conflict resolution
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={fetchAgents}
          title="Refresh agent data"
        >
          ↻
        </button>
      </div>

      {error && (
        <p style={{ color: "#dc2626", padding: "0 20px" }}>
          {error}
        </p>
      )}

      {agents.length === 0 ? (
        <div
          style={{
            padding: "35px",
            textAlign: "center",
            color: "#6b7280",
          }}
        >
          No agent trust information is available yet.
        </div>
      ) : (
        <div style={{ padding: "16px 20px" }}>
          {agents.map((agent) => {
            const color =
              AGENT_COLORS[agent.agent_id] || "#6b7280"

            const trustScore =
              Number(agent.trust_score ?? 0)

            const trustPercent = Math.round(
              trustScore <= 1
                ? trustScore * 100
                : trustScore
            )

            const totalWrites =
              Number(agent.total_writes ?? 0)

            const correctWrites =
              Number(agent.correct_writes ?? 0)

            const overturnedWrites =
              Number(agent.overturned_writes ?? 0)

            const accuracy =
              totalWrites > 0
                ? Math.round(
                    (correctWrites / totalWrites) * 100
                  )
                : 0

            return (
              <div
                key={agent.agent_id}
                style={{
                  marginBottom: "16px",
                  padding: "15px",
                  background: "#f8fafc",
                  borderRadius: "8px",
                  borderLeft: `4px solid ${color}`,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "10px",
                  }}
                >
                  <strong style={{ color }}>
                    {formatAgentName(agent.agent_id)}
                  </strong>

                  <strong
                    style={{
                      fontSize: "20px",
                      color,
                    }}
                  >
                    {trustPercent}%
                  </strong>
                </div>

                <div
                  style={{
                    height: "8px",
                    background: "#e5e7eb",
                    borderRadius: "6px",
                    overflow: "hidden",
                    marginBottom: "12px",
                  }}
                >
                  <div
                    style={{
                      width: `${Math.min(
                        trustPercent,
                        100
                      )}%`,
                      height: "100%",
                      background: color,
                    }}
                  />
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(4, minmax(0, 1fr))",
                    gap: "8px",
                  }}
                >
                  <div>
                    <strong>{totalWrites}</strong>
                    <small
                      style={{
                        display: "block",
                        color: "#6b7280",
                      }}
                    >
                      Total Writes
                    </small>
                  </div>

                  <div>
                    <strong>{correctWrites}</strong>
                    <small
                      style={{
                        display: "block",
                        color: "#6b7280",
                      }}
                    >
                      Correct
                    </small>
                  </div>

                  <div>
                    <strong>{overturnedWrites}</strong>
                    <small
                      style={{
                        display: "block",
                        color: "#6b7280",
                      }}
                    >
                      Overturned
                    </small>
                  </div>

                  <div>
                    <strong>{accuracy}%</strong>
                    <small
                      style={{
                        display: "block",
                        color: "#6b7280",
                      }}
                    >
                      Accuracy
                    </small>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}