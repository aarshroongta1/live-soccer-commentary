/**
 * Where the Python runtime lives, and how to hand one of its long-lived
 * streams straight through to the browser.
 *
 * A `rewrites()` entry is not enough for the two streaming endpoints: the dev
 * proxy negotiates gzip with the browser, and a gzipped `text/event-stream`
 * buffers until the encoder flushes — which, for a stream that emits a few
 * hundred bytes every couple of seconds, is never. Curl saw events; the page
 * saw a connection and silence. So these two go through route handlers that
 * copy the upstream body verbatim and mark it `no-transform`.
 */

export const API_ORIGIN = process.env.COMMENTARY_API_ORIGIN ?? "http://127.0.0.1:8000";

export async function proxyStream(
  request: Request,
  path: string,
  headers: Record<string, string>,
): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${API_ORIGIN}${path}`, {
      headers: { accept: headers["content-type"] ?? "*/*" },
      cache: "no-store",
      // Closing the tab must close the subscription, or the bus accumulates
      // queues for readers that went away.
      signal: request.signal,
    });
  } catch {
    return new Response("runtime unreachable", { status: 502 });
  }

  if (!upstream.ok || upstream.body === null) {
    return new Response("runtime unreachable", { status: 502 });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      ...headers,
      "cache-control": "no-cache, no-store, no-transform",
      "x-accel-buffering": "no",
    },
  });
}
