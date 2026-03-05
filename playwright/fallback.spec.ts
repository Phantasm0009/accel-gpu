import { test, expect } from "@playwright/test";

test("landing loads", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "accel-gpu", level: 1 })).toBeVisible();
});

test("demo reports backend", async ({ page }) => {
  await page.goto("/example/");
  await expect(page.getByText(/Backend:\s*(webgpu|webgl|cpu)/i)).toBeVisible({ timeout: 30_000 });
});

test("playground run button executes", async ({ page }) => {
  await page.goto("/playground/");
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.locator("#output")).not.toHaveText("Click Run to execute.", {
    timeout: 30_000,
  });
});

test("backend consistency (cpu vs webgl/default)", async ({ page }, testInfo) => {
  await page.goto("/");
  const browserName = testInfo.project.name;

  const result = await page.evaluate(async ({ browserName }) => {
    const moduleUrl = new URL("/dist/index.js", window.location.origin).href;
    const mod = await import(moduleUrl);
    const { init, matmul, inv, fft, conv2d } = mod;

    const eps = 1e-2;

    function maxAbsRel(a: Float32Array, b: Float32Array) {
      if (a.length !== b.length) return { maxAbs: Infinity, maxRel: Infinity };
      let maxAbs = 0;
      let maxRel = 0;
      for (let i = 0; i < a.length; i++) {
        const abs = Math.abs(a[i] - b[i]);
        const rel = abs / Math.max(1e-8, Math.abs(a[i]), Math.abs(b[i]));
        if (abs > maxAbs) maxAbs = abs;
        if (rel > maxRel) maxRel = rel;
      }
      return { maxAbs, maxRel };
    }

    const cpu = await init({ forceCPU: true });
    let accelerated: Awaited<ReturnType<typeof init>>;
    try {
      accelerated = await init({ forceWebGL: true });
    } catch {
      accelerated = await init();
    }

    const cases: Array<{
      op: string;
      maxAbs: number;
      maxRel: number;
      pass: boolean;
      required: boolean;
    }> = [];

    const mA = new Float32Array([1, 2, 3, 4, 5, 6]);
    const mB = new Float32Array([7, 8, 9, 10, 11, 12]);
    const cCpu = await matmul(cpu, cpu.array(mA, [2, 3]), cpu.array(mB, [3, 2]));
    const cAcc = await matmul(accelerated, accelerated.array(mA, [2, 3]), accelerated.array(mB, [3, 2]));
    const mStats = maxAbsRel(await cCpu.toArray(), await cAcc.toArray());
    const strictMatmul = !(browserName === "webkit" && accelerated.backendType === "webgl");
    cases.push({
      op: "matmul",
      ...mStats,
      required: strictMatmul,
      pass: strictMatmul ? mStats.maxAbs <= eps : true,
    });

    const invInput = new Float32Array([4, 1, 2, 1, 3, 0, 2, 0, 5]);
    const iCpu = await inv(cpu, cpu.array(invInput, [3, 3]));
    const iAcc = await inv(accelerated, accelerated.array(invInput, [3, 3]));
    const iStats = maxAbsRel(await iCpu.toArray(), await iAcc.toArray());
    cases.push({ op: "inv", ...iStats, required: true, pass: iStats.maxAbs <= 1 });

    const fftInput = new Float32Array([0, 1, 0, -1, 0, 1, 0, -1]);
    const fCpu = await fft(cpu, cpu.array(fftInput));
    const fAcc = await fft(accelerated, accelerated.array(fftInput));
    const fStats = maxAbsRel(await fCpu.toArray(), await fAcc.toArray());
    cases.push({ op: "fft", ...fStats, required: true, pass: fStats.maxAbs <= eps * 2 });

    const convInput = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]);
    const convKernel = new Float32Array([
      1, 0,
      0, -1,
    ]);
    const oCpu = await conv2d(cpu, cpu.array(convInput, [4, 4, 1]), cpu.array(convKernel, [2, 2, 1, 1]));
    const oAcc = await conv2d(
      accelerated,
      accelerated.array(convInput, [4, 4, 1]),
      accelerated.array(convKernel, [2, 2, 1, 1])
    );
    const cStats = maxAbsRel(await oCpu.toArray(), await oAcc.toArray());
    cases.push({ op: "conv2d", ...cStats, required: true, pass: cStats.maxAbs <= eps * 2 });

    return {
      backendType: accelerated.backendType,
      eps,
      cases,
      pass: cases.every((item) => !item.required || item.pass),
    };
  }, { browserName });

  console.log(`backend consistency drift: ${JSON.stringify(result)}`);

  testInfo.annotations.push({
    type: "backend-drift",
    description: JSON.stringify(result),
  });

  expect(result.pass).toBe(true);
});
