import { proxyStream } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export const revalidate = 0;

/** The agent event stream, passed through uncompressed. */
export function GET(request: Request): Promise<Response> {
  return proxyStream(request, "/api/stream", {
    "content-type": "text/event-stream; charset=utf-8",
    connection: "keep-alive",
  });
}
