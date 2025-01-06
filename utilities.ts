export async function makeRequest(
  url: string,
  type: string,
  headers: Record<string, string>,
  body: any
) {
  try {
    const response = await fetch(url, {
      method: type,
      headers,
      body:
        body && (type === "POST" || type === "PUT")
          ? JSON.stringify(body)
          : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return {
      status: response.status,
      data: await response.text(),
      headers: Object.fromEntries(response.headers),
    };
  } catch (error) {
    console.error("Error making request:", error);
    throw error;
  }
}
