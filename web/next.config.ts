import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // The floating dev badge lands on top of the rail's account chip.
  devIndicators: false,
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
