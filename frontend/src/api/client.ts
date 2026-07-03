import createClient from "openapi-fetch";
import type { paths } from "./schema";

/**
 * Typed API client for the Gridiron Edge REST API.
 *
 * Base URL points at the local FastAPI dev server. Change via
 * VITE_API_BASE_URL environment variable when a deploy story exists.
 *
 * Usage:
 *   const { data, error } = await apiClient.GET("/games/{game_id}", {
 *     params: { path: { game_id: "2026_01_KC_LAC" } },
 *   });
 */
export const apiClient = createClient<paths>({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
});
