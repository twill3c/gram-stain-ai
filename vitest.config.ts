import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    // 前処理の照合は 224x224x3 を全画素なめるので既定の 5 秒では足りない
    testTimeout: 120_000,
  },
});
