import { useState } from "react"
import "./App.css"

const facts = [
  {
    id: 1,
    incident: "INC0000001",
    type: "priority",
    value: "2-High",
    agent: "Intake Agent",
    confidence: 88,
    status: "Active",
    updated: "10:24 AM",
  },
  {
    id: 2,
    incident: "INC0000001",
    type: "state",
    value: "New",
    agent: "Intake Agent",
    confidence: 90,
    status: "Active",
    updated: "10:20 AM",
  },
  {
    id: 3,
    incident: "INC0000045",
    type: "state",
    value: "Resolved",
    agent: "Delivery Agent",
    confidence: 78,
    status: "Active",
    updated: "10:18 AM",
  },
  {
    id: 4,
    incident: "INC0000001",
    type: "priority",
    value: "3-Medium",
    agent: "Billing/Ops Agent",
    confidence: 61,
    status: "Contested",
    updated: "10:15 AM",
  },
  {
    id: 5,
    incident: "INC0000072",
    type: "assignment_group",
    value: "Network Team",
    agent: "Coordinator Agent",
    confidence: 85,
    status: "Active",
    updated: "10:10 AM",
  },
  {
    id: 6,
    incident: "INC0000045",
    type: "state",
    value: "Reopened",
    agent: "Billing/Ops Agent",
    confidence: 72,
    status: "Contested",
    updated: "10:05 AM",
  },
]

const activities = [
  "Intake Agent added a priority fact for INC0000001",
  "Billing/Ops Agent submitted a conflicting value",
  "Delivery Agent confirmed the state of INC0000045",
  "Coordinator Agent updated an assignment group",
]

function App() {
  const [incidentFilter, setIncidentFilter] = useState("All")
  const [agentFilter, setAgentFilter] = useState("All")
  const [showInput, setShowInput] = useState(false)
  const [message, setMessage] = useState("")
  const [history, setHistory] = useState([])

  const filteredFacts = facts.filter((fact) => {
    const incidentMatches =
      incidentFilter === "All" || fact.incident === incidentFilter

    const agentMatches =
      agentFilter === "All" || fact.agent === agentFilter

    return incidentMatches && agentMatches
  })

  const sendMessage = () => {
    if (!message.trim()) return

    setHistory([
      ...history,
      {
        id: Date.now(),
        text: message,
      },
    ])

    setMessage("")
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <h2>MAMS</h2>
            <span>IT Incident System</span>
          </div>
        </div>

        <nav className="navigation">
          <button className="nav-item active">▦ Overview</button>
          <button className="nav-item">▤ Memory State</button>
          <button className="nav-item">⚠ Conflicts</button>
          <button className="nav-item">♙ Agents</button>
          <button className="nav-item">▣ Review Queue</button>
          <button className="nav-item">⌁ Analytics</button>
          <button className="nav-item">◇ System Health</button>
          <button className="nav-item">⚙ Settings</button>
        </nav>

        <div className="profile">
          <div className="avatar">S</div>
          <div>
            <strong>Sneha</strong>
            <span>Person C · Frontend</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>Multi-Agent Shared Memory System</h1>
            <p>
              Confidence-aware memory, contradiction detection and conflict
              resolution
            </p>
          </div>

          <div className="topbar-actions">
            <span className="online-status">● System Online</span>
            <span className="updated-time">Last updated: 10:24 AM</span>

            <button
              className="input-button"
              onClick={() => setShowInput(true)}
            >
              + Upload / Input
            </button>

            <button className="refresh-button">↻</button>
          </div>
        </header>

        <section className="dashboard-content">
          <div className="stats-grid">
            <article className="stat-card blue">
              <span>Total Facts</span>
              <strong>1,248</strong>
              <small>Static Week 1 demo</small>
            </article>

            <article className="stat-card green">
              <span>Active Facts</span>
              <strong>1,102</strong>
              <small>88.3% of stored facts</small>
            </article>

            <article className="stat-card orange">
              <span>Contested Facts</span>
              <strong>38</strong>
              <small>Require conflict handling</small>
            </article>

            <article className="stat-card purple">
              <span>Average Confidence</span>
              <strong>86%</strong>
              <small>Across all source agents</small>
            </article>
          </div>

          <div className="dashboard-grid">
            <div className="left-column">
              <section className="panel filter-panel">
                <label>
                  Search incident
                  <input placeholder="Search by incident ID..." />
                </label>

                <label>
                  Incident
                  <select
                    value={incidentFilter}
                    onChange={(event) =>
                      setIncidentFilter(event.target.value)
                    }
                  >
                    <option value="All">All Incidents</option>
                    <option value="INC0000001">INC0000001</option>
                    <option value="INC0000045">INC0000045</option>
                    <option value="INC0000072">INC0000072</option>
                  </select>
                </label>

                <label>
                  Fact type
                  <select>
                    <option>All Types</option>
                    <option>priority</option>
                    <option>state</option>
                    <option>assignment_group</option>
                    <option>urgency</option>
                    <option>category</option>
                  </select>
                </label>

                <label>
                  Source agent
                  <select
                    value={agentFilter}
                    onChange={(event) => setAgentFilter(event.target.value)}
                  >
                    <option value="All">All Agents</option>
                    <option value="Intake Agent">Intake Agent</option>
                    <option value="Delivery Agent">Delivery Agent</option>
                    <option value="Billing/Ops Agent">
                      Billing/Ops Agent
                    </option>
                    <option value="Coordinator Agent">
                      Coordinator Agent
                    </option>
                  </select>
                </label>
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Memory State</h2>
                    <p>Current facts stored by the agent system</p>
                  </div>
                  <span className="conflict-alert">
                    ⚠ 2 conflicts detected
                  </span>
                </div>

                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Incident ID</th>
                        <th>Fact Type</th>
                        <th>Value</th>
                        <th>Source Agent</th>
                        <th>Confidence</th>
                        <th>Status</th>
                        <th>Updated</th>
                      </tr>
                    </thead>

                    <tbody>
                      {filteredFacts.map((fact) => (
                        <tr
                          key={fact.id}
                          className={
                            fact.status === "Contested"
                              ? "contested-row"
                              : ""
                          }
                        >
                          <td className="incident-id">{fact.incident}</td>
                          <td>{fact.type}</td>
                          <td>{fact.value}</td>
                          <td>
                            <span className="agent-badge">
                              {fact.agent}
                            </span>
                          </td>
                          <td>
                            <div className="confidence">
                              <span>{fact.confidence}%</span>
                              <div className="confidence-track">
                                <div
                                  className={`confidence-fill ${
                                    fact.confidence >= 80
                                      ? "high"
                                      : fact.confidence >= 60
                                        ? "medium"
                                        : "low"
                                  }`}
                                  style={{
                                    width: `${fact.confidence}%`,
                                  }}
                                />
                              </div>
                            </div>
                          </td>
                          <td>
                            <span
                              className={`status-badge ${fact.status.toLowerCase()}`}
                            >
                              {fact.status}
                            </span>
                          </td>
                          <td>{fact.updated}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Review Queue</h2>
                    <p>Conflicts that require human attention</p>
                  </div>
                  <span className="queue-count">2 pending</span>
                </div>

                <div className="review-list">
                  <div className="review-item">
                    <div>
                      <strong>INC0000001 · priority</strong>
                      <p>2-High versus 3-Medium</p>
                    </div>
                    <span className="confidence-gap">27% gap</span>
                    <button>Review</button>
                  </div>

                  <div className="review-item">
                    <div>
                      <strong>INC0000045 · state</strong>
                      <p>Resolved versus Reopened</p>
                    </div>
                    <span className="confidence-gap">6% gap</span>
                    <button>Review</button>
                  </div>
                </div>
              </section>
            </div>

            <aside className="right-column">
              <section className="panel side-panel">
                <h2>Recent Activity</h2>

                <div className="activity-list">
                  {activities.map((activity, index) => (
                    <div className="activity-item" key={activity}>
                      <span className={`activity-dot dot-${index}`} />
                      <div>
                        <p>{activity}</p>
                        <small>{10 + index}:2{index} AM</small>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel side-panel">
                <h2>Agent Overview</h2>

                <div className="agent-overview">
                  <div className="donut">
                    <div>
                      <strong>1,248</strong>
                      <span>Total Facts</span>
                    </div>
                  </div>

                  <ul>
                    <li>
                      <span className="legend blue-dot" />
                      Intake Agent <strong>32%</strong>
                    </li>
                    <li>
                      <span className="legend green-dot" />
                      Delivery Agent <strong>28%</strong>
                    </li>
                    <li>
                      <span className="legend orange-dot" />
                      Billing/Ops <strong>22%</strong>
                    </li>
                    <li>
                      <span className="legend purple-dot" />
                      Coordinator <strong>18%</strong>
                    </li>
                  </ul>
                </div>
              </section>

              <section className="panel side-panel">
                <h2>System Health</h2>

                <div className="health-list">
                  <p>
                    Dashboard <span className="health-online">● Online</span>
                  </p>
                  <p>
                    Memory Service{" "}
                    <span className="health-pending">● Pending</span>
                  </p>
                  <p>
                    Agent Pipeline{" "}
                    <span className="health-pending">● Pending</span>
                  </p>
                  <p>
                    Database{" "}
                    <span className="health-pending">● Pending</span>
                  </p>
                </div>
              </section>
            </aside>
          </div>
        </section>
      </main>

      {showInput && (
        <div className="drawer-overlay">
          <aside className="input-drawer">
            <div className="drawer-header">
              <div>
                <h2>Upload or Add Incident</h2>
                <p>Week 1 uses static input simulation</p>
              </div>

              <button
                className="close-button"
                onClick={() => setShowInput(false)}
              >
                ×
              </button>
            </div>

            <label>
              Processing agent
              <select>
                <option>Intake Agent</option>
                <option>Delivery Agent</option>
                <option>Billing/Ops Agent</option>
                <option>Coordinator Agent</option>
              </select>
            </label>

            <label>
              Source
              <select>
                <option>manual_input</option>
                <option>ticket_intake.csv</option>
                <option>monitoring_logs.csv</option>
                <option>field_reports.csv</option>
              </select>
            </label>

            <label>
              Incident update
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Example: INC0000001 has priority 2-High"
                rows="5"
              />
            </label>

            <label className="upload-area">
              <span>Upload CSV file</span>
              <input type="file" accept=".csv" />
            </label>

            <button className="send-button" onClick={sendMessage}>
              Send Input
            </button>

            <div className="input-history">
              <h3>Input History</h3>

              {history.length === 0 ? (
                <p className="empty-history">No input sent yet.</p>
              ) : (
                history.map((item) => (
                  <div className="history-message" key={item.id}>
                    <small>Manual input</small>
                    <p>{item.text}</p>
                  </div>
                ))
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}

export default App