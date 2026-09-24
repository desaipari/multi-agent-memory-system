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

  const [agentFilter, setAgentFilter] = useState("")
  const [eventFilter, setEventFilter] = useState("")
  const [entityFilter, setEntityFilter] = useState("")

  const [dateFrom, setDateFrom] = useState("")
  const [dateTo, setDateTo] = useState("")

  // Time range dropdown
  const [timeRange, setTimeRange] = useState("")

  const fetchLogs = useCallback(async () => {
    try {
      setError(null)

      const data = await memoryApi.getAuditLog({
        limit: 100,
        agent_id: agentFilter,
        event_type: eventFilter,
        entity: entityFilter,
        date_from: dateFrom,
        date_to: dateTo,
      })

      setLogs(data.logs || [])
    } catch (err) {
      console.error("Cannot fetch audit log:", err)
      setError("Unable to load audit trail.")
    } finally {
      setLoading(false)
    }
  }, [agentFilter, eventFilter, entityFilter, dateFrom, dateTo])

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

      {/* Filters */}
      <div
        style={{
          padding: "14px 20px",
          borderBottom: "1px solid #e5e7eb",
        }}
      >
        {/* Row 1 - Agent, Event and Entity */}
        <div
          style={{
            display: "flex",
            gap: "12px",
            alignItems: "center",
            marginBottom: "14px",
            flexWrap: "wrap",
          }}
        >
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value)}
            style={{
              flex: 1,
              minWidth: "220px",
              padding: "8px 10px",
              border: "1px solid #d1d5db",
              borderRadius: "6px",
            }}
          >
            <option value="">All Agents</option>
            <option value="intake_agent">Intake Agent</option>
            <option value="billing_agent">Billing Agent</option>
            <option value="delivery_agent">Delivery Agent</option>
            <option value="coordinator_agent">
              Coordinator Agent
            </option>
          </select>

          <select
            value={eventFilter}
            onChange={(e) => setEventFilter(e.target.value)}
            style={{
              flex: 1,
              minWidth: "220px",
              padding: "8px 10px",
              border: "1px solid #d1d5db",
              borderRadius: "6px",
            }}
          >
            <option value="">All Events</option>
            <option value="write">Write</option>
            <option value="corroboration">Corroboration</option>
            <option value="conflict_detected">
              Conflict Detected
            </option>
            <option value="auto_resolved">
              Auto Resolved
            </option>
            <option value="human_resolved">
              Human Resolved
            </option>
            <option value="action_blocked">
              Action Blocked
            </option>
          </select>

          <input
            type="text"
            placeholder="Incident ID (e.g. INC0000099)"
            value={entityFilter}
            onChange={(e) => setEntityFilter(e.target.value)}
            style={{
              flex: 1,
              minWidth: "220px",
              padding: "8px 10px",
              border: "1px solid #d1d5db",
              borderRadius: "6px",
              boxSizing: "border-box",
            }}
          />
        </div>

        {/* Row 2 - Time Range, From Date and To Date */}
        <div
          style={{
            display: "flex",
            gap: "20px",
            alignItems: "flex-end",
            flexWrap: "wrap",
          }}
        >
          {/* Time Range */}
          <div
            style={{
              flex: 1,
              minWidth: "250px",
            }}
          >
            <label
              style={{
                display: "block",
                marginBottom: "6px",
                fontSize: "12px",
                fontWeight: "500",
                color: "#374151",
              }}
            >
              Time Range
            </label>

            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 10px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                boxSizing: "border-box",
                backgroundColor: "white",
              }}
            >
              <option value="">All Time</option>
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
              <option value="1year">Last 1 Year</option>
              <option value="custom">Custom Date Range</option>
            </select>
          </div>

          {/* From Date */}
          <div
            style={{
              flex: 1,
              minWidth: "250px",
            }}
          >
            <label
              style={{
                display: "block",
                marginBottom: "6px",
                fontSize: "12px",
                fontWeight: "500",
                color: "#374151",
              }}
            >
              From Date
            </label>

            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 10px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                boxSizing: "border-box",
              }}
            />
          </div>

          {/* To Date */}
          <div
            style={{
              flex: 1,
              minWidth: "250px",
            }}
          >
            <label
              style={{
                display: "block",
                marginBottom: "6px",
                fontSize: "12px",
                fontWeight: "500",
                color: "#374151",
              }}
            >
              To Date
            </label>

            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 10px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                boxSizing: "border-box",
              }}
            />
          </div>
        </div>
      </div>

      {error && (
        <p
          style={{
            color: "#dc2626",
            padding: "0 20px",
          }}
        >
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