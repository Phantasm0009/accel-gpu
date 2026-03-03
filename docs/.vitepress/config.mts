import { defineConfig } from "vitepress";

export default defineConfig({
  title: "accel-gpu",
  description: "NumPy for the browser GPU",
  base: "/accel-gpu/",
  head: [["link", { rel: "icon", href: "/accel-gpu/icon.png" }]],
  themeConfig: {
    nav: [
      { text: "Quick Start", link: "/guide/quickstart" },
      { text: "API", link: "/api" },
      { text: "Examples", link: "https://phantasm0009.github.io/accel-gpu/example/" },
      { text: "Playground", link: "https://phantasm0009.github.io/accel-gpu/playground/" },
    ],
    sidebar: [
      {
        text: "Getting Started",
        items: [
          { text: "Overview", link: "/" },
          { text: "Quick Start", link: "/guide/quickstart" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "API Reference", link: "/api" },
        ],
      },
    ],
    search: {
      provider: "local",
    },
    socialLinks: [{ icon: "github", link: "https://github.com/Phantasm0009/accel-gpu" }],
    footer: {
      message: "MIT Licensed",
      copyright: "Copyright © 2026 accel-gpu contributors",
    },
    outline: "deep",
  },
});
