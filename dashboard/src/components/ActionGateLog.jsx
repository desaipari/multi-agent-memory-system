import { useCallback, useEffect, useState } from "react"
import { memoryApi } from "../api"

export default function ActionGateLog({ refreshTrigger = 0 }) {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchLogs = useCallback(async () => {
    try {
      setError(null)

      const data = await memoryApi.getActionGateLog()
      setLogs(data.logs || [])
    } catch (err) {
      console.error("Cannot fetch action gate log:", err)
      setError("Unable to load action gate log.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs, refreshTrigger])

  useEffect(() => {
    const interval = setInterval(fetchLogs, 5000)

    return () => clearInterval(interval)
  }, [fetchLogs])

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Action Gate Log</h2>
            <p>Loading action gate data...</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Action Gate Log</h2>

          <p>
            Actions blocked because memory confidence was too low
            or a fact was contested
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={fetchLogs}
          title="Refresh action gate log"
        >
          ↻
        </button>
      </div>

      {error && (
        <p style={{ color: "#dc2626", padding: "0 20px" }}>
          {error}
        </p>
      )}

      {logs.length === 0 ? (
        <div
          style={{
            padding: "35px",
            textAlign: "center",
            color: "#6b7280",
          }}
        >
          No actions have been blocked yet.
        </div>
      ) : (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Incident</th>
                <th>Fact Type</th>
                <th>Action Attempted</th>
                <th>Blocked Reason</th>
                <th>Confidence</th>
                <th>Time</th>
              </tr>
            </thead>

            <tbody>
              {logs.map((log, index) => (
                <tr key={log.gate_id || index}>
                  <td>
                    <span className="agent-badge">
                      {log.agent_id || "Unknown"}
                    </span>
                  </td>

                  <td className="incident-id">
                    {log.entity || "-"}
                  </td>

                  <td>{log.fact_type || "-"}</td>

                  <td>
                    {log.action_attempted || "-"}
                  </td>

                  <td>
                    {log.blocked_reason || "-"}
                  </td>

                  <td>
                    {log.confidence_at_block !== null &&
                    log.confidence_at_block !== undefined
                      ? `${Math.round(
                          log.confidence_at_block * 100
                        )}%`
                      : "-"}
                  </td>

                  <td>
                    {log.timestamp
                      ? new Date(
                          log.timestamp
                        ).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}