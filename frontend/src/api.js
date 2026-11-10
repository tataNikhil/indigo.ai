const API_URL = import.meta.env.VITE_API_URL || "";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// FastAPI returns errors as {detail: "..."} or, for validation errors,
// {detail: [{msg: "..."}, ...]}. Turn either into one readable sentence.
function readError(body, status) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail.map((d) => d.msg).join(", ");
  return `Request failed (${status})`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError("Can't reach the backend. Is uvicorn running on port 8000?", 0);
  }

  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(readError(body, response.status), response.status);
  return body;
}

// Leave sessionId empty to start a new conversation; the reply carries the new ID.
export function sendMessage(message, sessionId, channel = "chat") {
  const payload = { message, channel };
  if (sessionId) payload.session_id = sessionId;
  return request("/api/v1/chat", { method: "POST", body: JSON.stringify(payload) });
}

export function getSession(sessionId) {
  return request(`/api/v1/sessions/${sessionId}`);
}

export function deleteSession(sessionId) {
  return request(`/api/v1/sessions/${sessionId}`, { method: "DELETE" });
}
