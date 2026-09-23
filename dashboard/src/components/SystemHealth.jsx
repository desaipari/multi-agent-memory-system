import { useEffect, useState } from "react"
import { memoryApi } from "../api"

function SystemHealth() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const loadHealth = async () => {
    try {
      setLoading(true)
      setError("")

      const data = await memoryApi.checkHealth()
      setHealth(data)
    } catch (err) {
      console.error("Unable to load system health:", err)
      setHealth(null)
      setError("Unable to connect to the memory service.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadHealth()
  }, [])

  const backendOnline =
    health?.status === "running" ||
    health?.status === "ok" ||
    health?.status === "healthy"

  return (
    <section className="panel system-health-page">
      <div className="panel-heading">
        <div>
          <h2>System Health</h2>
          <p>
            Current status of the dashboard and memory infrastructure
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={loadHealth}
          title="Refresh system health"
        >
          ↻
        </button>
      </div>

      {loading ? (
        <p>Checking system health...</p>
      ) : (
        <>
          {error && (
            <p style={{ color: "red" }}>
              {error}
            </p>
          )}

          <div className="health-list">
            <p>
              Dashboard{" "}
              <span className="health-online">
                ● Online
              </span>
            </p>

            <p>
              Memory Service{" "}
              <span
                className={
                  backendOnline
                    ? "health-online"
                    : "health-pending"
                }
              >
                ● {backendOnline ? "Online" : "Offline"}
              </span>
            </p>

            <p>
              Qdrant{" "}
              <span
                className={
                  backendOnline
                    ? "health-online"
                    : "health-pending"
                }
              >
                ● {backendOnline ? "Running" : "Unknown"}
              </span>
            </p>

            <p>
              Database{" "}
              <span
                className={
                  backendOnline
                    ? "health-online"
                    : "health-pending"
                }
              >
                ● {backendOnline ? "Connected" : "Unknown"}
              </span>
            </p>
          </div>
        </>
      )}
    </section>
  )
}

export default SystemHealth