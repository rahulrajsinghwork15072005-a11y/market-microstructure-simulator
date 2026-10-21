import axios from "axios";

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "";
const api = axios.create({ baseURL: API_BASE });

export const getScenarios = () => api.get("/api/scenarios").then((r) => r.data);
export const getRun = (scenario: string) =>
  api.get("/api/demo/run", { params: { scenario } }).then((r) => r.data);
export const getStylized = (scenario: string) =>
  api.get("/api/demo/stylized-facts", { params: { scenario } }).then((r) => r.data);
export const getImpact = (scenario: string) =>
  api.get("/api/demo/impact", { params: { scenario } }).then((r) => r.data);
export const getBook = (scenario: string) =>
  api.get("/api/demo/book", { params: { scenario } }).then((r) => r.data);
