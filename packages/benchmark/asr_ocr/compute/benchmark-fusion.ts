import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const WER_PY = resolve(REPO_ROOT, 'packages', 'benchmark', 'separate', 'compute', 'wer.py');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');
const RESULTS_BASE = resolve(__dirname, '..', 'results');

interface Segment {
	text: string;
	start: number;
	end: number;
}

interface SegmentEx extends Segment {
	source: 'ocr' | 'asr';
}

function loadGT(): Segment[] {
	return JSON.parse(readFileSync(GROUND_TRUTH, 'utf-8')).result.segments;
}

function loadASR(): SegmentEx[] {
	const raw = JSON.parse(
		readFileSync(
			join(REPO_ROOT, 'packages', 'benchmark', 'asr', 'whisper', 'results', 'ggml-s1-sidechain', 'baseline', 'metadata', 'whisper_raw.json'),
			'utf-8',
		),
	);
	return raw.transcription.map((s: any) => ({
		text: s.text,
		start: s.offsets.from,
		end: s.offsets.to,
		source: 'asr' as const,
	}));
}

function loadOCR(): SegmentEx[] {
	const raw = JSON.parse(
		readFileSync(
			join(REPO_ROOT, 'packages', 'benchmark', 'ocr', 'results', 'ocr-cpp-fps1-so-ts0.3', 'metadata', 'asr.json'),
			'utf-8',
		),
	);
	return raw.result.segments.map((s: any) => ({
		text: s.text,
		start: s.start,
		end: s.end,
		source: 'ocr' as const,
	}));
}

function calcCERDetail(hypText: string, gtSegs: Segment[]): any {
	const refText = gtSegs.map((s) => s.text).join('');
	const r = spawnSync('python3', [WER_PY, '--text', refText, hypText], {
		timeout: 15_000,
		encoding: 'utf-8',
	});
	if (r.status !== 0) throw new Error(`wer.py failed: ${r.stderr}`);
	return JSON.parse(r.stdout);
}

function segmentsToText(segs: Segment[]): string {
	return segs.map((s) => s.text).join('');
}

function mergeSameText(segs: Segment[]): Segment[] {
	const merged: Segment[] = [];
	for (const s of segs) {
		if (!s.text) continue;
		const last = merged[merged.length - 1];
		if (last && last.text === s.text && s.start - last.end <= 1000) {
			last.end = Math.max(last.end, s.end);
		} else {
			merged.push({ ...s });
		}
	}
	return merged;
}

// Strategy 1: OCR baseline
function fusionOCR(ocr: SegmentEx[]): Segment[] {
	return mergeSameText(ocr.sort((a, b) => a.start - b.start));
}

// Strategy 2: ASR baseline
function fusionASR(asr: SegmentEx[]): Segment[] {
	return mergeSameText(asr.sort((a, b) => a.start - b.start));
}

// Strategy 3: Split OCR segments by ASR transitions
// For each OCR segment, check if ASR has a segment inside with different text.
// If so, split OCR segment and insert ASR segment.
function fusionSplit(ocr: SegmentEx[], asr: SegmentEx[]): Segment[] {
	const sortedOCR = [...ocr].sort((a, b) => a.start - b.start);
	const sortedASR = [...asr].sort((a, b) => a.start - b.start);
	const result: Segment[] = [];

	for (const ocrSeg of sortedOCR) {
		const asrInside = sortedASR.filter(
			(a) => a.start >= ocrSeg.start && a.end <= ocrSeg.end && a.text !== ocrSeg.text,
		);

		if (asrInside.length === 0) {
			result.push({ text: ocrSeg.text, start: ocrSeg.start, end: ocrSeg.end });
			continue;
		}

		// Split OCR segment at ASR boundaries
		let cursor = ocrSeg.start;
		for (const asrSeg of asrInside) {
			if (asrSeg.start > cursor) {
				result.push({ text: ocrSeg.text, start: cursor, end: asrSeg.start });
			}
			result.push({ text: asrSeg.text, start: asrSeg.start, end: asrSeg.end });
			cursor = asrSeg.end;
		}
		if (cursor < ocrSeg.end) {
			result.push({ text: ocrSeg.text, start: cursor, end: ocrSeg.end });
		}
	}

	return mergeSameText(result);
}

// Strategy 4: OCR + ASR union — use ASR segments where OCR has no text
// in that timeframe, or ASR segment is short and OCR segment is long
function fusionUnionWithSplit(ocr: SegmentEx[], asr: SegmentEx[]): Segment[] {
	const sortedOCR = [...ocr].sort((a, b) => a.start - b.start);
	const sortedASR = [...asr].sort((a, b) => a.start - b.start);

	// For each ASR segment, check if it has an OCR segment covering its time
	// If not covered, add ASR segment
	const result: SegmentEx[] = [...sortedOCR];

	for (const asrSeg of sortedASR) {
		const ocrCovering = sortedOCR.find(
			(o) => o.start <= asrSeg.start && o.end >= asrSeg.end - 500,
		);
		if (!ocrCovering) {
			result.push(asrSeg);
		}
	}

	return mergeSameText(result.sort((a, b) => a.start - b.start));
}

// Strategy 5: ASR segments < 2s that differ from OCR → use ASR text
// Short ASR segments are likely rapid-fire dialogue that OCR misses
function fusionShortASR(ocr: SegmentEx[], asr: SegmentEx[], maxDurMs: number = 2000): Segment[] {
	const sortedOCR = [...ocr].sort((a, b) => a.start - b.start);
	const result: SegmentEx[] = [...sortedOCR];

	for (const asrSeg of asr) {
		const dur = asrSeg.end - asrSeg.start;
		if (dur > maxDurMs) continue;

		// Check if OCR has this segment
		const ocrMatch = sortedOCR.some((o) => {
			if (o.start > asrSeg.end || o.end < asrSeg.start) return false;
			return o.text.includes(asrSeg.text) || asrSeg.text.includes(o.text);
		});
		if (!ocrMatch) {
			result.push(asrSeg);
		}
	}

	return mergeSameText(result.sort((a, b) => a.start - b.start));
}

function run(label: string, segs: Segment[], asrSegs: SegmentEx[], ocrSegs: SegmentEx[], desc: string) {
	const text = segmentsToText(segs);
	const cerDetail = calcCERDetail(text, loadGT());

	const gtSegs = loadGT();
	const hypText = text;
	const refText = gtSegs.map((s) => s.text).join('');

	const resultDir = join(RESULTS_BASE, label);
	mkdirSync(resultDir, { recursive: true });
	mkdirSync(join(resultDir, 'metadata'), { recursive: true });

	const summary = {
		label,
		desc,
		segments: segs.length,
		hyp_chars: hypText.length,
		ref_chars: cerDetail.ref_chars,
		cer: cerDetail.cer,
		wer: cerDetail.wer,
		char_subs: cerDetail.char_subs ?? 0,
		char_ins: cerDetail.char_ins ?? 0,
		char_dels: cerDetail.char_dels ?? 0,
	};

	writeFileSync(join(resultDir, 'metadata', 'summary.json'), JSON.stringify(summary, null, 2));

	// Write source data for provenance
	const srcTime = (s: SegmentEx) => ({
		text: s.text,
		start: s.start,
		end: s.end,
		start_fmt: fmtTime(s.start),
		end_fmt: fmtTime(s.end),
	});

	writeFileSync(join(resultDir, 'metadata', 'asr.json'), JSON.stringify({
		_engine: 'whisper.cpp',
		_model: 'ggml-large-v3-turbo',
		_device: 'Vulkan (RADV)',
		_source: 'packages/benchmark/asr/whisper/results/ggml-s1-sidechain/baseline/metadata/whisper_raw.json',
		result: {
			text: segmentsToText(asrSegs),
			segments: asrSegs.filter((s) => s.source === 'asr').map(srcTime),
		},
	}, null, 2));

	writeFileSync(join(resultDir, 'metadata', 'ocr.json'), JSON.stringify({
		_engine: 'cpp-ort',
		_model: 'rapidocr (det+cls+rec)',
		_device: 'CPU',
		_source: 'packages/benchmark/ocr/results/ocr-cpp-fps1-so-ts0.3/metadata/asr.json',
		result: {
			text: segmentsToText(ocrSegs),
			segments: ocrSegs.filter((s) => s.source === 'ocr').map(srcTime),
		},
	}, null, 2));

	const outFile = label.includes('asr-only') ? 'asr.json'
		: label.includes('ocr-only') ? 'ocr.json'
		: 'asr_ocr.json';

	const baseMeta = {
		result: {
			text: hypText,
			segments: segs.map((s) => ({
				text: s.text,
				start: s.start,
				end: s.end,
				start_fmt: fmtTime(s.start),
				end_fmt: fmtTime(s.end),
			})),
		},
	};

	let output: any;
	if (outFile === 'asr.json') {
		output = {
			_engine: 'whisper.cpp',
			_model: 'ggml-large-v3-turbo',
			_device: 'Vulkan (RADV)',
			_params: { fps: 1, textScore: 0.3, subtitleOnly: true },
			_input: 'packages/benchmark/ref/media/video_source.mp4',
			...baseMeta,
		};
	} else if (outFile === 'ocr.json') {
		output = {
			_engine: 'cpp-ort',
			_model: 'rapidocr (det+cls+rec)',
			_device: 'CPU',
			_params: { fps: 1, textScore: 0.3, subtitleOnly: true },
			_source: 'packages/benchmark/ocr/results/ocr-cpp-fps1-so-ts0.3/metadata/asr.json',
			...baseMeta,
		};
	} else {
		output = {
			_engine: 'asr_ocr',
			_strategy: label.replace('asr-ocr-fusion-', ''),
			_asr: {
				engine: 'whisper.cpp',
				model: 'ggml-large-v3-turbo',
				device: 'Vulkan (RADV)',
				source: 'packages/benchmark/asr/whisper/results/ggml-s1-sidechain/baseline/metadata/whisper_raw.json',
			},
			_ocr: {
				engine: 'cpp-ort',
				model: 'rapidocr (det+cls+rec)',
				device: 'CPU',
				source: 'packages/benchmark/ocr/results/ocr-cpp-fps1-so-ts0.3/metadata/asr.json',
			},
			_fusion_params: {
				minConfidence: 0.3,
			},
			...baseMeta,
		};
	}

	writeFileSync(join(resultDir, 'metadata', outFile), JSON.stringify(output, null, 2));

	console.log(`[${label}] ${desc}`);
	console.log(`  segs=${segs.length} hyp=${hypText.length} ref=${cerDetail.ref_chars}`);
	console.log(`  CER=${(cerDetail.cer * 100).toFixed(2)}% WER=${(cerDetail.wer * 100).toFixed(2)}%`);
	console.log(`  S=${cerDetail.char_subs} I=${cerDetail.char_ins} D=${cerDetail.char_dels}`);
	console.log();
}

function fmtTime(ms: number): string {
	const h = Math.floor(ms / 3600000);
	const m = Math.floor((ms % 3600000) / 60000);
	const s = Math.floor((ms % 60000) / 1000);
	const ml = ms % 1000;
	return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ml.toString().padStart(3, '0')}`;
}

if (require.main === module) {
	const gtSegs = loadGT();
	const asrSegs = loadASR();
	const ocrSegs = loadOCR();

	console.log('=== Data summary ===');
	console.log(`GT:  ${gtSegs.length} segs, ${gtSegs.reduce((a, s) => a + s.text.length, 0)} chars`);
	console.log(`ASR: ${asrSegs.length} segs, ${asrSegs.reduce((a, s) => a + s.text.length, 0)} chars`);
	console.log(`OCR: ${ocrSegs.length} segs, ${ocrSegs.reduce((a, s) => a + s.text.length, 0)} chars`);
	console.log();

	const runs: { label: string; fn: () => Segment[]; desc: string }[] = [
		{
			label: 'asr-ocr-fusion-ocr-only',
			fn: () => fusionOCR(ocrSegs),
			desc: 'OCR baseline',
		},
		{
			label: 'asr-ocr-fusion-asr-only',
			fn: () => fusionASR(asrSegs),
			desc: 'ASR baseline',
		},
		{
			label: 'asr-ocr-fusion-split',
			fn: () => fusionSplit(ocrSegs, asrSegs),
			desc: 'OCR split by ASR transitions',
		},
		{
			label: 'asr-ocr-fusion-union',
			fn: () => fusionUnionWithSplit(ocrSegs, asrSegs),
			desc: 'OCR + unfilled ASR segments',
		},
		{
			label: 'asr-ocr-fusion-short-asr',
			fn: () => fusionShortASR(ocrSegs, asrSegs, 2000),
			desc: 'OCR + short ASR segs (<2s)',
		},
		{
			label: 'asr-ocr-fusion-short-asr-1500',
			fn: () => fusionShortASR(ocrSegs, asrSegs, 1500),
			desc: 'OCR + short ASR segs (<1.5s)',
		},
	];

	const results: any[] = [];
	for (const r of runs) {
		const segs = r.fn();
		run(r.label, segs, asrSegs, ocrSegs, r.desc);
		const s = JSON.parse(readFileSync(join(RESULTS_BASE, r.label, 'metadata', 'summary.json'), 'utf-8'));
		results.push(s);
	}

	console.log('======= FUSION BENCHMARK SUMMARY =======');
	console.log('label                      | segs | hyp_ch | ref_ch | CER%   | WER%   | S  | I  | D  |');
	console.log('---------------------------|------|--------|--------|--------|--------|----|----|----|');
	for (const r of results) {
		const l = r.label.padEnd(27);
		const seg = String(r.segments).padStart(4);
		const hc = String(r.hyp_chars).padStart(6);
		const rc = String(r.ref_chars).padStart(6);
		const cer = (r.cer * 100).toFixed(2).padStart(6);
		const wer = (r.wer * 100).toFixed(2).padStart(6);
		const s = String(r.char_subs).padStart(2);
		const i = String(r.char_ins).padStart(2);
		const d = String(r.char_dels).padStart(2);
		console.log(`${l} | ${seg} | ${hc} | ${rc} | ${cer}% | ${wer}% | ${s} | ${i} | ${d} |`);
	}
}
