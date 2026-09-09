import * as ort from 'onnxruntime-node';
import { join, resolve } from 'node:path';
import { existsSync, readdirSync } from 'node:fs';
import { REPO_ROOT } from '@repo/config/root';


/** Find rapidocr model directory dynamically. */
function findRapidOCRModelsDir(): string {
	const venvBase = resolve(REPO_ROOT, '.venv');
	const sitePackagesBase = resolve(venvBase, process.platform === 'win32' ? 'Lib' : 'lib', 'site-packages');
	if (!existsSync(sitePackagesBase)) {
		throw new Error(`site-packages not found: ${sitePackagesBase}`);
	}
	let spDir = sitePackagesBase;
	if (process.platform !== 'win32') {
		const entries = readdirSync(spDir, { withFileTypes: true });
		const pyDir = entries.find(e => e.isDirectory() && e.name.startsWith('python'));
		if (pyDir) {
			spDir = join(spDir, pyDir.name);
		}
	}
	const modelsDir = join(spDir, 'rapidocr_onnxruntime', 'models');
	if (!existsSync(modelsDir)) {
		throw new Error(`rapidocr models not found: ${modelsDir}`);
	}
	return modelsDir;
}

const MODEL_DIR = findRapidOCRModelsDir();
const DET_PATH = join(MODEL_DIR, 'ch_PP-OCRv3_det_infer.onnx');

async function main() {
	console.log('=== OCR WebGPU Feasibility Test ===\n');
	console.log('Backends:', ort.listSupportedBackends());

	// 1. Load det model with webgpu
	console.log('\n--- Loading detection model with webgpu...');
	let t0 = performance.now();
	let session: ort.InferenceSession;
	let usedProvider: string;
	try {
		session = await ort.InferenceSession.create(DET_PATH, {
			executionProviders: ['webgpu'],
		});
		usedProvider = 'webgpu';
	} catch (e) {
		console.log('  webgpu FAILED:', (e as Error).message);
		console.log('  Falling back to cpu...');
		session = await ort.InferenceSession.create(DET_PATH, {
			executionProviders: ['cpu'],
		});
		usedProvider = 'cpu';
	}
	const loadMs = performance.now() - t0;
	console.log(`  OK (${usedProvider}) — Load: ${loadMs.toFixed(1)}ms`);
	console.log(`  Inputs: ${session.inputNames?.join(', ') || 'N/A'}`);
	console.log(`  Outputs: ${session.outputNames?.join(', ') || 'N/A'}`);

	// 2. Dummy input (736x736, normalized)
	const H = 736, W = 736;
	const data = new Float32Array(3 * H * W);
	for (let c = 0; c < 3; c++) {
		const mean = [0.485, 0.456, 0.406][c];
		const std = [0.229, 0.224, 0.225][c];
		for (let i = 0; i < H * W; i++) {
			data[c * H * W + i] = (128 / 255 - mean) / std;
		}
	}
	const feed: Record<string, ort.Tensor> = {};
	feed[session.inputNames[0]] = new ort.Tensor('float32', data, [1, 3, H, W]);

	// 3. Warmup
	console.log('\n--- Warmup...');
	await session.run(feed);

	// 4. Timed inference (10 runs)
	console.log('\n--- Timed runs (10)...');
	const times: number[] = [];
	for (let i = 0; i < 10; i++) {
		t0 = performance.now();
		const out = await session.run(feed);
		const ms = performance.now() - t0;
		times.push(ms);
		const outTensor = out[session.outputNames[0]];
		console.log(`  Run ${i + 1}: ${ms.toFixed(1)}ms — shape: [${outTensor.dims?.join(',')}]`);
	}

	const avg = times.reduce((a, b) => a + b, 0) / times.length;
	console.log(`\n  Avg: ${avg.toFixed(1)}ms (${usedProvider})`);

	await session.release();

	// 5. Compare to CPU baseline for det only
	console.log('\n--- CPU baseline for same model...');
	const cpuSession = await ort.InferenceSession.create(DET_PATH, {
		executionProviders: ['cpu'],
	});
	const cpuFeed: Record<string, ort.Tensor> = {};
	cpuFeed[cpuSession.inputNames[0]] = new ort.Tensor('float32', data, [1, 3, H, W]);
	await cpuSession.run(cpuFeed); // warmup
	const cpuTimes: number[] = [];
	for (let i = 0; i < 10; i++) {
		t0 = performance.now();
		await cpuSession.run(cpuFeed);
		cpuTimes.push(performance.now() - t0);
	}
	const cpuAvg = cpuTimes.reduce((a, b) => a + b, 0) / cpuTimes.length;
	console.log(`  CPU avg: ${cpuAvg.toFixed(1)}ms`);
	console.log(`  Speedup: ${(cpuAvg / avg).toFixed(1)}x`);

	await cpuSession.release();
}

main().catch(console.error);
