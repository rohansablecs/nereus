const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = "Request failed";

    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {}

    throw new Error(detail);
  }

  return response.json();
}

export async function getDecisionMetadata() {
  return request("/api/decision/metadata");
}

export async function analyzeDecision(payload) {
  return request("/api/decision/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function simulateScenario(payload) {
  return request("/api/decision/scenario", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
