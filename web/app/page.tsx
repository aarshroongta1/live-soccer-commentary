import { Watch } from "@/components/Watch";

/**
 * `?mock=1` is decided here, on the server, rather than after hydration.
 *
 * It has to be: the page that comes back already contains the video element
 * and the toggle, so a mode chosen later would mean the browser requests the
 * MJPEG from a runtime that is not running, and the toggle renders with the
 * wrong button pressed until React catches up. In fixture mode nothing on
 * this page touches the network at all.
 */
export default async function Page({ searchParams }: PageProps<"/">) {
  const params = await searchParams;
  const mock = params.mock !== undefined;
  return <Watch initialSource={mock ? "fixture" : "live"} />;
}
