import { proxyStream } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export const revalidate = 0;

/** The delayed broadcast. The boundary must match `BOUNDARY` in the server app. */
export function GET(request: Request): Promise<Response> {
  return proxyStream(request, "/api/video", {
    "content-type": "multipart/x-mixed-replace; boundary=frame",
  });
}
