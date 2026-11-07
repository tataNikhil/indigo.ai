const API_URL = import.meta.env.VITE_API_URL || "";

// FastAPI returns errors as {detail: "..."} or, for validation errors,
// {detail: [{msg: "..."}, ...]}. Turn either into one readable sentence.
function readError(body, status) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail.map((d) => d.msg).join(", ");
  return `Request failed (${status})`;
}

export async function sendMessage(message, channel = "chat") {
  let response;
  try {
    response = await fetch(`${API_URL}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, channel }),
    });
  } catch {
    throw new Error("Can't reach the backend. Is uvicorn running on port 8000?");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(readError(body, response.status));
  return body;
}
