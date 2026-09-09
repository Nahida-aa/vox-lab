import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve, join } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..', '..');
const GROUND_TRUTH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'srt_manual.json');
const RESULTS_BASE = resolve(__dirname, '..', 'results');

interface Segment {
	text: string;
	start: number;
	end: number;
}

function loadSegments(path: string): Segment[] {
	const d = JSON.parse(readFileSync(path, 'utf-8'));
	return d.result.segments.map((s: any) => ({
		text: s.text.trim(),
		start: s.start / 1000,
		end: s.end / 1000,
	}));
}

function iou(a: Segment, b: Segment): number {
	const interStart = Math.max(a.start, b.start);
	const interEnd = Math.min(a.end, b.end);
	const intersection = Math.max(0, interEnd - interStart);
	const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
	return union > 0 ? intersection / union : 0;
}

interface OffsetResult {
	label: string;
	source: string;
	params: string;
	n_gt: number;
	n_hyp: number;
	start_offset_mean: number;
	start_offset_median: number;
	end_offset_mean: number;
	end_offset_median: number;
	detection_rate: number;        // GT segments detected (IOU>0 with any hyp)
	missed: Segment[];             // GT segments with IOU=0 against all hyp
	false_positive_count: number;  // hyp segments with IOU=0 against all GT
	hyp_char_count: number;
	ref_char_count: number;
}

function analyzeOffsets(gt: Segment[], hyp: Segment[], label: string, source: string, params: string): OffsetResult {
	// For each GT segment, find best-matching hyp segment
	const offsets_start: number[] = [];
	const offsets_end: number[] = [];
	const matchedHyp = new Set<number>();
	const missed: Segment[] = [];

	for (const g of gt) {
		let bestIou = 0;
		let bestIdx = -1;
		for (let i = 0; i < hyp.length; i++) {
			const o = iou(g, hyp[i]);
			if (o > bestIou) {
				bestIou = o;
				bestIdx = i;
			}
		}
		if (bestIou > 0) {
			matchedHyp.add(bestIdx);
			offsets_start.push(hyp[bestIdx].start - g.start);
			offsets_end.push(hyp[bestIdx].end - g.end);
		} else {
			missed.push(g);
		}
	}

	// false positives: hyp segments that match no GT
	let fpCount = 0;
	for (let i = 0; i < hyp.length; i++) {
		if (!matchedHyp.has(i)) {
			let hasOverlap = false;
			for (const g of gt) {
				if (iou(hyp[i], g) > 0) { hasOverlap = true; break; }
			}
			if (!hasOverlap) fpCount++;
		}
	}

	const sortedStart = [...offsets_start].sort((a, b) => a - b);
	const sortedEnd = [...offsets_end].sort((a, b) => a - b);
	const mid = Math.floor(sortedStart.length / 2);

	const gtText = gt.map(s => s.text).join('');
	const hypText = hyp.map(s => s.text).join('');

	return {
		label,
		source,
		params,
		n_gt: gt.length,
		n_hyp: hyp.length,
		start_offset_mean: offsets_start.length > 0
			? parseFloat((offsets_start.reduce((a, b) => a + b, 0) / offsets_start.length).toFixed(3))
			: 0,
		start_offset_median: sortedStart.length > 0
			? parseFloat((sortedStart.length % 2 === 1 ? sortedStart[mid] : (sortedStart[mid - 1] + sortedStart[mid]) / 2).toFixed(3))
			: 0,
		end_offset_mean: offsets_end.length > 0
			? parseFloat((offsets_end.reduce((a, b) => a + b, 0) / offsets_end.length).toFixed(3))
			: 0,
		end_offset_median: sortedEnd.length > 0
			? parseFloat((sortedEnd.length % 2 === 1 ? sortedEnd[mid] : (sortedEnd[mid - 1] + sortedEnd[mid]) / 2).toFixed(3))
			: 0,
		detection_rate: parseFloat((1 - missed.length / gt.length).toFixed(3)),
		missed,
		false_positive_count: fpCount,
		hyp_char_count: hypText.length,
		ref_char_count: gtText.length,
	};
}

function loadCER(source: string, param: string): { cer: number; wer: number } | null {
	try {
		const d = JSON.parse(readFileSync(
			join(RESULTS_BASE, source, param, 'metadata', 'summary.json'), 'utf-8'));
		return { cer: d.cer, wer: d.wer };
	} catch {
		return null;
	}
}

// --- Main ---
if (require.main === module) {
	const gt = loadSegments(GROUND_TRUTH);
	const gtText = gt.map(s => s.text).join('');

	const results: OffsetResult[] = [];
	const asrPaths: string[] = [];

	function walk(dir: string, base: string) {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			const full = join(dir, entry.name);
			if (entry.isDirectory()) {
				walk(full, base);
			} else if (entry.name === 'asr.json') {
				asrPaths.push(full.slice(base.length + 1));
			}
		}
	}
	walk(RESULTS_BASE, RESULTS_BASE);
	asrPaths.sort();

	for (const asrPath of asrPaths) {
		const parts = asrPath.split('/');
		const source = parts[0];
		const param = parts[1];
		const label = `${source}_${param}`;

		const hyp = loadSegments(join(RESULTS_BASE, asrPath));
		const r = analyzeOffsets(gt, hyp, label, source, param);
		results.push(r);
	}

	// Also load CER per result
	const cerMap = new Map<string, { cer: number; wer: number }>();
	for (const r of results) {
		const c = loadCER(r.source, r.params);
		if (c) cerMap.set(r.label, c);
	}

	// === SUMMARY TABLE ===
	console.log('\n=========== TIMESTAMP OFFSET + CER ===========');
	console.log('source          | param           | segs | det%  | FP | s_off_mn | s_off_md | e_off_mn | e_off_md | CER%   | hyp_ch');
	console.log('----------------|-----------------|------|-------|----|----------|----------|----------|----------|--------|-------');

	for (const r of results) {
		const src = r.source.padEnd(16);
		const p = r.params.padEnd(17);
		const seg = String(r.n_hyp).padStart(4);
		const det = (r.detection_rate * 100).toFixed(1).padStart(5);
		const fp = String(r.false_positive_count).padStart(3);
		const som = r.start_offset_mean.toFixed(3).padStart(8);
		const somd = r.start_offset_median.toFixed(3).padStart(8);
		const eom = r.end_offset_mean.toFixed(3).padStart(8);
		const eomd = r.end_offset_median.toFixed(3).padStart(8);

		const cer = cerMap.get(r.label);
	const cerStr = cer ? (cer.cer * 100).toFixed(2).padStart(6) : '  N/A ';
	const hypCh = String(r.hyp_char_count).padStart(7);

		const line = `${src} | ${p} | ${seg} | ${det}% | ${fp} | ${som} | ${somd} | ${eom} | ${eomd} | ${cerStr}% | ${hypCh}`;
		console.log(line);
	}

	// === MISSED SEGMENTS DETAIL ===
	console.log('\n=========== MISSED SEGMENTS ===========');
	console.log('(GT segments with zero IOU against all hyp segments)');
	// Only show params that have notable misses around 71s and 115s
	const targetRuns = results.filter(r =>
		r.missed.some(s => (Math.abs(s.start - 71.2) < 5) || (Math.abs(s.start - 115.4) < 5))
	);
	const shown = new Set<string>();
	for (const r of targetRuns) {
		if (shown.has(r.params)) continue;
		shown.add(r.params);
		const key = `${r.source}/${r.params}`;
		console.log(`\n--- ${key} ---`);
		for (const m of r.missed) {
			if (Math.abs(m.start - 71.2) > 5 && Math.abs(m.start - 115.4) > 5) continue;
			// Find closest hypothetic segment text
			console.log(`  [${m.start.toFixed(2)}-${m.end.toFixed(2)}] "${m.text}"`);
		}
	}

	// === OFFSET DIRECTION ANALYSIS ===
	console.log('\n=========== OFFSET DIRECTION (raw baseline for reference) ===========');
	// Show what offsets look like for key param combos
	const interestParams = ['baseline', 'temp-02', 'vad-v6', 'vad-v6-th02'];
	const interestSources = ['ggml-s1-raw', 'ggml-s1-sidechain'];
	for (const src of interestSources) {
		for (const p of interestParams) {
			const r = results.find(x => x.source === src && x.params === p);
			if (!r) continue;
			const cer = cerMap.get(r.label);
			const pct = ((cer?.cer ?? 0) * 100).toFixed(2);
			console.log(`\n${src}/${p} (CER=${pct}%)`);
			console.log(`  GT segments: ${r.n_gt}, Hyp segments: ${r.n_hyp}`);
			const detPct = (r.detection_rate * 100).toFixed(1);
			console.log(`  Detection rate: ${detPct}%`);
			const somSign = r.start_offset_mean >= 0 ? '+' : '';
			const somdSign = r.start_offset_median >= 0 ? '+' : '';
			const eomSign = r.end_offset_mean >= 0 ? '+' : '';
			const eomdSign = r.end_offset_median >= 0 ? '+' : '';
			console.log(`  Start offset mean=${somSign}${r.start_offset_mean.toFixed(3)}s median=${somdSign}${r.start_offset_median.toFixed(3)}s`);
			console.log(`  End offset   mean=${eomSign}${r.end_offset_mean.toFixed(3)}s median=${eomdSign}${r.end_offset_median.toFixed(3)}s`);
			console.log(`  False positives: ${r.false_positive_count}`);

			// Show offsets for the segments around 71s and 115s
			const gt = loadSegments(GROUND_TRUTH);
			const hyp = loadSegments(join(RESULTS_BASE, src, p, 'metadata', 'asr.json'));
			for (const g of gt) {
				if (Math.abs(g.start - 71.2) > 5 && Math.abs(g.start - 115.4) > 5) continue;
				let bestIou = 0, bestIdx = -1;
				for (let i = 0; i < hyp.length; i++) {
					const o = iou(g, hyp[i]);
					if (o > bestIou) { bestIou = o; bestIdx = i; }
				}
				if (bestIdx >= 0) {
					const h = hyp[bestIdx];
					const soff = h.start - g.start;
					const eoff = h.end - g.end;
					const soffSign = soff >= 0 ? '+' : '';
					const eoffSign = eoff >= 0 ? '+' : '';
					console.log(`  GT [${g.start.toFixed(2)}-${g.end.toFixed(2)}] "${g.text}"`);
					console.log(`    → hyp [${h.start.toFixed(2)}-${h.end.toFixed(2)}] "${h.text}" (s=${soffSign}${soff.toFixed(3)}s e=${eoffSign}${eoff.toFixed(3)}s iou=${bestIou.toFixed(2)})`);
				} else {
					console.log(`  GT [${g.start.toFixed(2)}-${g.end.toFixed(2)}] "${g.text}" → MISSED`);
				}
			}
		}
	}
}
