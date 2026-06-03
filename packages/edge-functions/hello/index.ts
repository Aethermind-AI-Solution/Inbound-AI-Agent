import { serve } from "https://deno.land/std@0.177.0/http/server.ts";

serve(async (_req: Request) => {
  return new Response(JSON.stringify({ status: "ok", service: "voice-booking-agent" }), {
    headers: { "Content-Type": "application/json" },
  });
});
