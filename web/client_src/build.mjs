import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const sourceDir = resolve(__dirname, "src");
const distDir = resolve(__dirname, "..", "..", "src", "blink", "web", "client_dist");

rmSync(distDir, { force: true, recursive: true });
mkdirSync(distDir, { recursive: true });
cpSync(sourceDir, distDir, { recursive: true });

console.log(`Copied Blink browser UI assets from ${sourceDir} to ${distDir}`);
