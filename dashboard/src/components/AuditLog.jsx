import { useCallback, useEffect, useState } from "react"
import { memoryApi } from "../api"

const EVENT_COLORS = {
  write: { bg: "#dbeafe", color: "#1e40af" },
  corroboration: { bg: "#dcfce7", color: "#166534" },
  conflict_detected: { bg: "#fee2e2", color: "#991b1b" },
  auto_resolved: { bg: "#d1fae5", color: "#065f46" },
  human_resolved: { bg: "#e0e7ff", color: "#3730a3" },
  action_blocked: { bg: "#fef3c7", color: "#92400e" },
}

export default function AuditLog({ refreshTrigger = 0 }) {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchLogs = useCallback(async () => {
    try {
      setError(null)

      const data = await memoryApi.getAuditLog(100)
      setLogs(data.logs || [])
    } catch (err) {
      console.error("Cannot fetch audit log:", err)
      setError("Unable to load audit trail.")
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
            <h2>Full Audit Trail</h2>
            <p>Loading audit events...</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Full Audit Trail</h2>

          <p>
            Every write, corroboration, conflict, resolution,
            and blocked action
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={fetchLogs}
          title="Refresh audit trail"
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
          No audit events recorded yet.
        </div>
      ) : (
        <div
          style={{
            maxHeight: "600px",
            overflowY: "auto",
          }}
        >
          {logs.map((log, index) => {
            const eventStyle =
              EVENT_COLORS[log.event_type] || {
                bg: "#f3f4f6",
                color: "#374151",
              }

            return (
              <div
                key={log.log_id || index}
                style={{
                  padding: "11px 18px",
                  borderBottom: "1px solid #e5e7eb",
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "12px",
                }}
              >
                <span
                  style={{
                    backgroundColor: eventStyle.bg,
                    color: eventStyle.color,
                    padding: "3px 9px",
                    borderRadius: "10px",
                    fontSize: "11px",
                    fontWeight: "600",
                    whiteSpace: "nowrap",
                  }}
                >
                  {log.event_type || "event"}
                </span>

                <span
                  style={{
                    flex: 1,
                    fontSize: "13px",
                    color: "#374151",
                  }}
                >
                  {log.description || "No description"}
                </span>

                <span
                  style={{
                    fontSize: "11px",
                    color: "#9ca3af",
                    whiteSpace: "nowrap",
                  }}
                >
                  {log.timestamp
                    ? new Date(
                        log.timestamp
                      ).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "-"}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}