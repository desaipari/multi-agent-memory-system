import { useState, useRef } from "react"
import axios from "axios"

const BASE_URL = "http://localhost:8000"

const AGENTS = [
  { id: "intake",      label: "Intake Agent",      color: "#7c3aed" },
  { id: "billing",     label: "Billing/Ops Agent", color: "#dc2626" },
  { id: "coordinator", label: "Coordinator Agent", color: "#059669" }
]

const SOURCE_FILES = [
  "ticket_intake.csv",
  "monitoring_logs.csv",
  "field_reports.csv",
  "manual_input"
]

export default function InputPanel({ onInputSent }) {
  const [text, setText] = useState("")
  const [selectedAgent, setSelectedAgent] = useState("intake")
  const [selectedSource, setSelectedSource] = useState("manual_input")
  const [chatHistory, setChatHistory] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const fileInputRef = useRef(null)

  const currentAgent = AGENTS.find(a => a.id === selectedAgent)

  const addMessage = (type, text, agent = null) => {
    setChatHistory(prev => [...prev, {
      id: Date.now() + Math.random(),
      type,
      text,
      agent,
      timestamp: new Date().toLocaleTimeString()
    }])
  }

  const handleSendText = async () => {
    if (!text.trim() || isLoading) return

    const messageText = text
    setText("")
    addMessage("user", messageText, selectedAgent)
    setIsLoading(true)

    try {
      const response = await axios.post(`${BASE_URL}/chat`, {
        message: messageText,
        agent_role: selectedAgent,
        source_file: selectedSource
      }, { timeout: 30000 })

      const data = response.data
      let responseText = ""

      if (data.extracted_fact) {
        const f = data.extracted_fact
        responseText = (
          `[${selectedAgent}_agent] Extracted: ` +
          `${f.fact_type}=${f.value} ` +
          `(confidence: ${(f.confidence * 100).toFixed(0)}%)`
        )
      } else if (data.conflicts_resolved !== undefined) {
        responseText = (
          `[coordinator_agent] Resolved ` +
          `${data.conflicts_resolved} conflicts`
        )
      } else if (data.error) {
        responseText = `[error] ${data.error}`
      } else {
        responseText = `[${selectedAgent}_agent] ${data.status || "processed"}`
      }

      if (data.contradiction_detected) {
        const details = data.contradiction_details || {}
        responseText += ` | ⚠ CONTRADICTION: ${details.resolution || "detected"}`
      }

      if (data.recommendation) {
        responseText += ` | Recommendation generated`
      }

      addMessage("agent", responseText, selectedAgent)
      if (onInputSent) onInputSent()

    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      addMessage("system", `Error: ${errorMsg}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Reset file input so same file can be re-uploaded
    if (fileInputRef.current) fileInputRef.current.value = ""

    addMessage("system", `Uploading: ${file.name}...`)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append("file", file)
    formData.append("agent_role", selectedAgent)

    try {
      const response = await axios.post(
        `${BASE_URL}/upload`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 180000,
          onUploadProgress: (e) => {
            const pct = Math.round((e.loaded * 100) / e.total)
            setUploadProgress(pct)
          }
        }
      )

      const data = response.data
      const processed = data.rows_processed || data.sentences_processed || 0
      const contradictions = data.contradictions_found || 0

      addMessage(
        "agent",
        `[${selectedAgent}_agent] Processed ${file.name}: ` +
        `${processed} facts written, ` +
        `${contradictions} contradictions found`
      )
      if (onInputSent) onInputSent()

    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      addMessage("system", `Upload failed: ${errorMsg}`)
    } finally {
      setUploadProgress(null)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendText()
    }
  }

  return (
    <div style={{
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
      display: "flex",
      flexDirection: "column",
      height: "100%"
    }}>
      {/* Header */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid #e5e7eb" }}>
        <h2 style={{ margin: "0 0 10px 0", fontSize: "15px" }}>
          Input Panel
        </h2>

        {/* Agent selector */}
        <div style={{ marginBottom: "8px" }}>
          <label style={{ fontSize: "11px", color: "#6b7280", display: "block", marginBottom: "4px" }}>
            Active Agent
          </label>
          <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
            {AGENTS.map(agent => (
              <button
                key={agent.id}
                onClick={() => setSelectedAgent(agent.id)}
                style={{
                  padding: "4px 10px",
                  borderRadius: "12px",
                  border: "2px solid",
                  borderColor: selectedAgent === agent.id ? agent.color : "#e5e7eb",
                  backgroundColor: selectedAgent === agent.id ? agent.color : "white",
                  color: selectedAgent === agent.id ? "white" : "#6b7280",
                  fontSize: "11px",
                  cursor: "pointer",
                  fontWeight: selectedAgent === agent.id ? "600" : "400"
                }}
              >
                {agent.label}
              </button>
            ))}
          </div>
        </div>

        {/* Source selector */}
        <div>
          <label style={{ fontSize: "11px", color: "#6b7280", display: "block", marginBottom: "4px" }}>
            Source
          </label>
          <select
            value={selectedSource}
            onChange={e => setSelectedSource(e.target.value)}
            style={{
              width: "100%",
              padding: "5px 8px",
              borderRadius: "4px",
              border: "1px solid #d1d5db",
              fontSize: "12px"
            }}
          >
            {SOURCE_FILES.map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Chat History */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        padding: "10px",
        display: "flex",
        flexDirection: "column",
        gap: "6px"
      }}>
        {chatHistory.length === 0 && (
          <div style={{
            textAlign: "center",
            color: "#9ca3af",
            fontSize: "12px",
            marginTop: "30px",
            lineHeight: "1.6"
          }}>
            Type an incident update or upload a CSV file
            <br />
            e.g. "INC0000001 has priority 2-High"
          </div>
        )}
        {chatHistory.map(msg => (
          <div key={msg.id} style={{
            padding: "7px 10px",
            borderRadius: "6px",
            backgroundColor:
              msg.type === "user" ? "#eff6ff"
              : msg.type === "agent" ? "#f0fdf4"
              : "#fafafa",
            borderLeft: "3px solid",
            borderLeftColor:
              msg.type === "user" ? "#3b82f6"
              : msg.type === "agent" ? "#22c55e"
              : "#d1d5db",
            fontSize: "12px"
          }}>
            <div style={{ fontSize: "10px", color: "#9ca3af", marginBottom: "2px" }}>
              {msg.timestamp} {msg.agent ? `— ${msg.agent}` : ""}
            </div>
            {msg.text}
          </div>
        ))}

        {isLoading && (
          <div style={{
            padding: "7px 10px",
            backgroundColor: "#f9fafb",
            borderRadius: "6px",
            fontSize: "12px",
            color: "#9ca3af",
            animation: "pulse 1s infinite"
          }}>
            Processing...
          </div>
        )}
      </div>

      {/* Upload Progress */}
      {uploadProgress !== null && (
        <div style={{ padding: "8px 12px", borderTop: "1px solid #e5e7eb" }}>
          <div style={{
            height: "4px",
            backgroundColor: "#e5e7eb",
            borderRadius: "2px"
          }}>
            <div style={{
              width: `${uploadProgress}%`,
              height: "100%",
              backgroundColor: "#7c3aed",
              borderRadius: "2px",
              transition: "width 0.3s"
            }} />
          </div>
          <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "4px" }}>
            Uploading... {uploadProgress}%
          </div>
        </div>
      )}

      {/* Input Area */}
      <div style={{ padding: "10px", borderTop: "1px solid #e5e7eb" }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={`Type incident update for ${currentAgent?.label}...\ne.g. INC0000001 priority is 2-High`}
          rows={3}
          disabled={isLoading}
          style={{
            width: "100%",
            padding: "7px",
            borderRadius: "4px",
            border: "1px solid #d1d5db",
            fontSize: "12px",
            resize: "none",
            boxSizing: "border-box",
            fontFamily: "inherit",
            opacity: isLoading ? 0.6 : 1
          }}
        />
        <div style={{ display: "flex", gap: "6px", marginTop: "6px" }}>
          <label style={{
            padding: "7px 12px",
            backgroundColor: "#f3f4f6",
            border: "1px solid #d1d5db",
            borderRadius: "4px",
            cursor: isLoading ? "not-allowed" : "pointer",
            fontSize: "12px",
            color: "#374151",
            opacity: isLoading ? 0.6 : 1
          }}>
            📁 Upload CSV
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.pdf"
              onChange={handleFileUpload}
              disabled={isLoading}
              style={{ display: "none" }}
            />
          </label>
          <button
            onClick={handleSendText}
            disabled={!text.trim() || isLoading}
            style={{
              flex: 1,
              padding: "7px",
              backgroundColor:
                text.trim() && !isLoading
                  ? currentAgent?.color
                  : "#e5e7eb",
              color:
                text.trim() && !isLoading ? "white" : "#9ca3af",
              border: "none",
              borderRadius: "4px",
              cursor: text.trim() && !isLoading ? "pointer" : "not-allowed",
              fontSize: "12px",
              fontWeight: "600"
            }}
          >
            {isLoading ? "Processing..." : "Send →"}
          </button>
        </div>
      </div>
    </div>
  )
}