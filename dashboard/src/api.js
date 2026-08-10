import axios from "axios";

const BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
});

export const memoryApi = {
  getAllFacts: async () => {
    const response = await api.get("/memory/all");
    return response.data;
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
      source_file: sourceFile,
    });

    return response.data;
  },

  checkHealth: async () => {
    const response = await api.get("/memory/health");
    return response.data;
  },
};