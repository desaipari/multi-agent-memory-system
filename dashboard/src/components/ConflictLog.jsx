import { useCallback, useEffect, useState } from "react"
import { memoryApi } from "../api"

const AGENT_COLORS = {
  intake_agent: "#7c3aed",
  delivery_agent: "#2563eb",
  billing_agent: "#dc2626",
  billing_ops_agent: "#dc2626",
  coordinator_agent: "#059669",
}

export default function ConflictLog({
  refreshTrigger = 0,
  onResolved,
}) {
  const [conflicts, setConflicts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [resolvingId, setResolvingId] = useState(null)

  const fetchConflicts = useCallback(async () => {
    try {
      setError(null)

      const data = await memoryApi.getConflicts()
      setConflicts(data.conflicts || [])
    } catch (err) {
      console.error("Cannot fetch conflicts:", err)
      setError("Unable to load conflict data.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchConflicts()
  }, [fetchConflicts, refreshTrigger])

  useEffect(() => {
    const interval = setInterval(fetchConflicts, 5000)

    return () => clearInterval(interval)
  }, [fetchConflicts])

  const resolveConflict = async (
    conflict,
    winningFactId,
    winningValue
  ) => {
    const reason = window.prompt(
      `Why should "${winningValue}" be selected as the correct value?`,
      "Selected after human review"
    )

    if (reason === null) return

    try {
      setResolvingId(conflict.conflict_id)

      await memoryApi.resolveConflict(
        conflict.conflict_id,
        winningFactId,
        "Sneha",
        reason
      )

      await fetchConflicts()

      if (onResolved) {
        onResolved()
      }
    } catch (err) {
      console.error("Conflict resolution failed:", err)
      window.alert(
        "Unable to resolve the conflict. Check the backend."
      )
    } finally {
      setResolvingId(null)
    }
  }

  const flagged = conflicts.filter(
    (conflict) => conflict.status === "flagged"
  )

  const resolved = conflicts.filter(
    (conflict) => conflict.status !== "flagged"
  )

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Conflict Log</h2>
            <p>Loading detected conflicts...</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
      }}
    >
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Conflict Log</h2>

            <p>
              Contradictory facts detected by the shared memory
              service
            </p>
          </div>

          <div
            style={{
              display: "flex",
              gap: "10px",
              alignItems: "center",
            }}
          >
            <span className="queue-count">
              {flagged.length} pending
            </span>

            <button
              className="refresh-button"
              onClick={fetchConflicts}
              title="Refresh conflicts"
            >
              ↻
            </button>
          </div>
        </div>

        {error && (
          <p style={{ color: "#dc2626", padding: "0 20px" }}>
            {error}
          </p>
        )}

        {flagged.length === 0 ? (
          <div
            style={{
              padding: "35px",
              textAlign: "center",
              color: "#6b7280",
            }}
          >
            No conflicts currently require human review.
          </div>
        ) : (
          <div
            style={{
              padding: "16px",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
            }}
          >
            {flagged.map((conflict) => (
              <div
                key={conflict.conflict_id}
                className="review-item"
                style={{
                  display: "block",
                  background: "#fff7ed",
                  padding: "16px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "12px",
                  }}
                >
                  <div>
                    <strong>
                      {conflict.entity} · {conflict.fact_type}
                    </strong>

                    <p style={{ margin: "4px 0 0" }}>
                      Human review required
                    </p>
                  </div>

                  <span className="confidence-gap">
                    {conflict.confidence_gap !== null &&
                    conflict.confidence_gap !== undefined
                      ? `${Math.round(
                          conflict.confidence_gap * 100
                        )}% gap`
                      : "Gap unavailable"}
                  </span>
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "12px",
                  }}
                >
                  <div
                    style={{
                      background: "white",
                      border: "1px solid #e5e7eb",
                      borderRadius: "8px",
                      padding: "12px",
                    }}
                  >
                    <strong>{conflict.value_a}</strong>

                    <p
                      style={{
                        color:
                          AGENT_COLORS[conflict.agent_a] ||
                          "#6b7280",
                        margin: "5px 0",
                      }}
                    >
                      {conflict.agent_a}
                    </p>

                    <small>
                      Confidence:{" "}
                      {conflict.confidence_a !== null &&
                      conflict.confidence_a !== undefined
                        ? `${Math.round(
                            conflict.confidence_a * 100
                          )}%`
                        : "N/A"}
                    </small>

                    <button
                      onClick={() =>
                        resolveConflict(
                          conflict,
                          conflict.fact_id_a,
                          conflict.value_a
                        )
                      }
                      disabled={
                        resolvingId === conflict.conflict_id
                      }
                      style={{
                        display: "block",
                        marginTop: "10px",
                      }}
                    >
                      {resolvingId === conflict.conflict_id
                        ? "Resolving..."
                        : "Select as Correct"}
                    </button>
                  </div>

                  <div
                    style={{
                      background: "white",
                      border: "1px solid #e5e7eb",
                      borderRadius: "8px",
                      padding: "12px",
                    }}
                  >
                    <strong>{conflict.value_b}</strong>

                    <p
                      style={{
                        color:
                          AGENT_COLORS[conflict.agent_b] ||
                          "#6b7280",
                        margin: "5px 0",
                      }}
                    >
                      {conflict.agent_b}
                    </p>

                    <small>
                      Confidence:{" "}
                      {conflict.confidence_b !== null &&
                      conflict.confidence_b !== undefined
                        ? `${Math.round(
                            conflict.confidence_b * 100
                          )}%`
                        : "N/A"}
                    </small>

                    <button
                      onClick={() =>
                        resolveConflict(
                          conflict,
                          conflict.fact_id_b,
                          conflict.value_b
                        )
                      }
                      disabled={
                        resolvingId === conflict.conflict_id
                      }
                      style={{
                        display: "block",
                        marginTop: "10px",
                      }}
                    >
                      {resolvingId === conflict.conflict_id
                        ? "Resolving..."
                        : "Select as Correct"}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Resolved Conflicts</h2>

            <p>
              Automatically or manually resolved contradictions
            </p>
          </div>

          <span className="queue-count">
            {resolved.length} resolved
          </span>
        </div>

        {resolved.length === 0 ? (
          <div
            style={{
              padding: "30px",
              textAlign: "center",
              color: "#6b7280",
            }}
          >
            No resolved conflicts yet.
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Incident</th>
                  <th>Fact Type</th>
                  <th>Value A</th>
                  <th>Value B</th>
                  <th>Resolution</th>
                  <th>Reason</th>
                </tr>
              </thead>

              <tbody>
                {resolved.map((conflict) => (
                  <tr key={conflict.conflict_id}>
                    <td className="incident-id">
                      {conflict.entity}
                    </td>

                    <td>{conflict.fact_type}</td>

                    <td>{conflict.value_a || "-"}</td>

                    <td>{conflict.value_b || "-"}</td>

                    <td>
                      <span className="status-badge active">
                        {conflict.status === "auto_resolved"
                          ? "Auto-resolved"
                          : "Human-resolved"}
                      </span>
                    </td>

                    <td>
                      {conflict.resolution_reason || "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}