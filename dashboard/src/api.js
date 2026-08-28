import axios from "axios"

const BASE_URL = "http://localhost:8000"

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000
})

export const memoryApi = {
  getAllFacts: async () => {
    const response = await api.get("/memory/all")
    return response.data
  },

  writeFact: async (
    entity,
    factType,
    value,
    agentId,
    extractionType,
    confidence,
    sourceFile
  ) => {
    const response = await api.post("/memory/write", {
      entity,
      fact_type: factType,
      value,
      agent_id: agentId,
      extraction_type: extractionType,
      confidence,
      source_file: sourceFile
    })

    return response.data
  },

  getConflicts: async () => {
    const response = await api.get("/memory/conflicts")
    return response.data
  },

  getAgents: async () => {
    const response = await api.get("/memory/agents")
    return response.data
  },

  getAuditLog: async (limit = 50) => {
    const response = await api.get(`/memory/audit?limit=${limit}`)
    return response.data
  },

  getActionGateLog: async () => {
    const response = await api.get("/memory/action_gate_log")
    return response.data
  },

  resolveConflict: async (
    conflictId,
    winningFactId,
    resolvedBy,
    reason
  ) => {
    const response = await api.post("/memory/resolve", {
      conflict_id: conflictId,
      winning_fact_id: winningFactId,
      resolved_by: resolvedBy,
      reason
    })

    return response.data
  },

  checkAction: async (
    agentId,
    entity,
    factType,
    action,
    threshold
  ) => {
    const response = await api.post("/memory/check_action", {
      agent_id: agentId,
      entity,
      fact_type: factType,
      action_attempted: action,
      confidence_threshold: threshold || 0.60
    })

    return response.data
  },

  checkHealth: async () => {
    const response = await api.get("/memory/health")
    return response.data
  }
}