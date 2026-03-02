/**
 * Build site for GitHub Pages - copies static files + dist to deploy folder
 */
import { cpSync, mkdirSync, rmSync, existsSync } from "fs";
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
copy("example");
copy("benchmark");
copy("playground");
copy("dist");

console.log("Site built to deploy/");
