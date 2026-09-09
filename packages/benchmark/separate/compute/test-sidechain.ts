import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const PYTHON_BIN = join(REPO_ROOT, '.venv', 'bin', 'python');
const WER_PY = resolve(__dirname, 'wer.py');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');
const WHISPER_CLI = join(REPO_ROOT, 'submodule', 'whisper.cpp', 'build', 'bin', 'whisper-vulkan');
const MODEL = process.env.WHISPER_MODEL || join(REPO_ROOT, 'data', 'models', 'whisper', 'ggml-large-v3-turbo.bin');

const RESULTS_DIR = resolve(REPO_ROOT, 'packages', 'benchmark', 'separate', 'results', 'ggml-s1-sidechain');
const MEDIA_DIR = resolve(REPO_ROOT, 'packages', 'benchmark', 'separate', 'results', 'ggml-s1', 'media');

interface SidechainParams {
	threshold: number;
	ratio: number;
	attack: number;
	release: number;
	reduceBgm: number; // dB
}

function paramLabel(p: SidechainParams): string {
	return `sc_t${p.threshold}_r${p.ratio}_a${p.attack}_rel${p.release}_bgm${p.reduceBgm}`;
}

function ffmpegMix(p: SidechainParams, vocalsPath: string, bgmPath: string, outPath: string) {
	const sc = `threshold=${p.threshold}:ratio=${p.ratio}:attack=${p.attack}:release=${p.release}`;
	const filter = p.reduceBgm !== 0
		? `[0:a]asplit[v][v_key];[1:a][v_key]sidechaincompress=${sc}[bgm_sc];[bgm_sc]volume=${p.reduceBgm}dB[bgm_final];[v][bgm_final]amix=inputs=2:duration=first:weights=1 1[out]`
		: `[0:a]asplit[v][v_key];[1:a][v_key]sidechaincompress=${sc}[bgm_sc];[v][bgm_sc]amix=inputs=2:duration=first:weights=1 1[out]`;
	const r = spawnSync('ffmpeg', [
		'-y', '-i', vocalsPath, '-i', bgmPath,
		'-filter_complex', filter,
		'-map', '[out]', outPath,
	], { timeout: 60_000 });
	if (r.status !== 0) throw new Error(`ffmpeg failed: ${r.stderr?.toString().slice(-200)}`);
}

function whisperTranscribe(audioPath: string): any {
	const r = spawnSync(WHISPER_CLI, [
		'-m', MODEL, audioPath, '-l', 'zh', '-t', '4', '-ojf',
	], { timeout: 600_000 });
	if (r.status !== 0 && r.status !== null) {
		throw new Error(`whisper-vulkan exit ${r.status}: ${r.stderr?.toString().slice(-300)}`);
	}
	const jsonPath = `${audioPath}.json`;
	if (!existsSync(jsonPath)) throw new Error(`whisper did not produce ${jsonPath}`);
	return JSON.parse(readFileSync(jsonPath, 'utf-8'));
}

function computeCER(asrPath: string): any {
	const r = spawnSync(PYTHON_BIN, [WER_PY, GROUND_TRUTH, asrPath], {
		timeout: 30_000,
		env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
	});
	if (r.status !== 0) throw new Error(`wer.py failed: ${r.stderr?.toString().slice(-300)}`);
	return JSON.parse(r.stdout.toString());
}

function runOne(p: SidechainParams) {
	const label = paramLabel(p);
	const outDir = join(RESULTS_DIR, label);
	const mediaOut = join(outDir, 'media');
	const metadataOut = join(outDir, 'metadata');
	mkdirSync(mediaOut, { recursive: true });
	mkdirSync(metadataOut, { recursive: true });

	const vocals = join(MEDIA_DIR, 'target_3_vocals.wav');
	const bgm = join(MEDIA_DIR, 'target_bgm.wav');
	const mixedPath = join(mediaOut, 'target_3_vocals_mixed.wav');

	console.log(`\n=== ${label} ===`);

	// Step 1: mix
	const t0 = performance.now();
	ffmpegMix(p, vocals, bgm, mixedPath);
	console.log(`  mix: ${((performance.now() - t0) / 1000).toFixed(2)}s`);

	// Step 2: transcribe
	const t1 = performance.now();
	const raw = whisperTranscribe(mixedPath);
	const whisperElapsed = (performance.now() - t1) / 1000;
	console.log(`  whisper: ${whisperElapsed.toFixed(2)}s`);

	// Save raw whisper JSON
	writeFileSync(join(metadataOut, 'whisper_raw.json'), JSON.stringify(raw, null, 2));

	// Convert to pipeline format
	const segments: any[] = (Array.isArray(raw.transcription) ? raw.transcription : raw.transcription?.segments ?? [])
		.map((s: any) => ({
			text: (s.text || '').trim(),
			start: s.offsets?.from ?? 0,
			end: s.offsets?.to ?? 0,
			words: (s.tokens || [])
				.filter((t: any) => { const txt = t.text?.trim(); return txt && !txt.startsWith('[') && t.offsets?.from != null; })
				.map((t: any) => ({ word: t.text.trim(), start: t.offsets.from ?? 0, end: t.offsets.to ?? 0, probability: t.p })),
		}));
	const text = segments.map(s => s.text).join(' ');
	const audioDurMs = segments.length > 0 ? segments[segments.length - 1].end : 0;
	const rtf = whisperElapsed / Math.max(audioDurMs / 1000, 1);

	const asrOutput = {
		audio_info: { duration: audioDurMs },
		result: { text, segments },
		_engine: 'whisper.cpp',
		_device: 'vulkan',
		_rtf: rtf.toFixed(3),
		_params: p,
	};
	const asrPath = join(metadataOut, 'asr.json');
	writeFileSync(asrPath, JSON.stringify(asrOutput, null, 2));

	// Step 3: compute CER
	const t2 = performance.now();
	const result = computeCER(asrPath);
	console.log(`  cer: ${((performance.now() - t2) / 1000).toFixed(2)}s`);

	console.log(`  RTF: ${rtf.toFixed(3)} (${whisperElapsed.toFixed(1)}s / ${(audioDurMs / 1000).toFixed(1)}s)`);
	console.log(`  WER: ${(result.wer * 100).toFixed(2)}% | CER: ${(result.cer * 100).toFixed(2)}%`);
	console.log(`  Hyp chars: ${result.hyp_chars}, Ref chars: ${result.ref_chars}`);

	const summary = { label, params: p, rtf: Number(rtf.toFixed(3)), ...result };
	writeFileSync(join(metadataOut, 'summary.json'), JSON.stringify(summary, null, 2));
	return summary;
}

// --- Main ---
if (require.main === module) {
	const paramSets: SidechainParams[] = [
		// Baseline (current defaults)
		{ threshold: 0.1, ratio: 20, attack: 1, release: 500, reduceBgm: -12 },

		// Higher BGM mix
		{ threshold: 0.1, ratio: 20, attack: 1, release: 500, reduceBgm: -6 },
		{ threshold: 0.1, ratio: 20, attack: 1, release: 500, reduceBgm: -3 },
		{ threshold: 0.1, ratio: 20, attack: 1, release: 500, reduceBgm: 0 },

		// Lower threshold (compressor engages earlier → more BGM ducking → less BGM in mix)
		{ threshold: 0.05, ratio: 20, attack: 1, release: 500, reduceBgm: -12 },
		{ threshold: 0.01, ratio: 20, attack: 1, release: 500, reduceBgm: -12 },

		// Lower ratio (gentler compression → more BGM through)
		{ threshold: 0.1, ratio: 10, attack: 1, release: 500, reduceBgm: -12 },
		{ threshold: 0.1, ratio: 4, attack: 1, release: 500, reduceBgm: -12 },

		// Faster release (BGM recovers quicker between speech)
		{ threshold: 0.1, ratio: 20, attack: 1, release: 200, reduceBgm: -12 },
		{ threshold: 0.1, ratio: 20, attack: 1, release: 100, reduceBgm: -12 },
	];

	const results: any[] = [];
	for (const p of paramSets) {
		const label = paramLabel(p);
		if (existsSync(join(RESULTS_DIR, label, 'metadata', 'summary.json'))) {
			console.log(`[skip] ${label} already done`);
			results.push(JSON.parse(readFileSync(join(RESULTS_DIR, label, 'metadata', 'summary.json'), 'utf-8')));
			continue;
		}
		results.push(runOne(p));
	}

	console.log('\n======= SUMMARY =======');
	console.log('config                                       | RTF    | WER%   | CER%   | hyp_char | ref_char');
	console.log('---------------------------------------------|--------|--------|--------|----------|---------');
	for (const r of results) {
		const label = paramLabel(r.params).padEnd(44);
		const rtf = String(r.rtf.toFixed(3)).padStart(6);
		const wer = (r.wer * 100).toFixed(2).padStart(6);
		const cer = (r.cer * 100).toFixed(2).padStart(6);
		const hc = String(r.hyp_chars).padStart(8);
		const rc = String(r.ref_chars).padStart(8);
		console.log(`${label} | ${rtf} | ${wer}% | ${cer}% | ${hc} | ${rc}`);
	}
}
