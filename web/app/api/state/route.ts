import { API_ORIGIN } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export const revalidate = 0;

/**
 * The snapshot the page reads once, for the things that do not arrive as
 * events — how far behind the cursor the picture is being held, above all.
 *
 * Short and finite, so unlike the two streaming endpoints it is an ordinary
 * JSON round trip rather than a body copied through verbatim.
 */
export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(`${API_ORIGIN}/api/state`, { cache: "no-store" });
    if (!upstream.ok) return new Response("runtime unreachable", { status: 502 });
    return new Response(await upstream.text(), {
      status: 200,
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  } catch {
    return new Response("runtime unreachable", { status: 502 });
  }
}
