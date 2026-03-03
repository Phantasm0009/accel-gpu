/**
 * Build site for GitHub Pages - copies static files + dist to deploy folder
 */
import { cpSync, mkdirSync, rmSync, existsSync, writeFileSync } from "fs";
import { join } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const root = join(__dirname, "..");
const deploy = join(root, "deploy");

if (existsSync(deploy)) rmSync(deploy, { recursive: true });
mkdirSync(deploy, { recursive: true });

const copy = (src, dest = src) => {
  cpSync(join(root, src), join(deploy, dest), { recursive: true });
};

copy("index.html");
copy("icon.png");
copy("example");
copy("benchmark");
copy("playground");
copy("dist");

const docsDist = join(root, "docs", ".vitepress", "dist");
if (existsSync(docsDist)) {
  cpSync(docsDist, deploy, {
    recursive: true,
    filter: (src) => !src.endsWith(join(".vitepress", "dist", "index.html")),
  });
}

// Prevent Jekyll processing (ensures all files are served)
writeFileSync(join(deploy, ".nojekyll"), "");

console.log("Site built to deploy/");
