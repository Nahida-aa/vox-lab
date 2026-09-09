import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const REPO_ROOT = resolve(__dirname, '..', '..', '..', '..');
const GT_PATH = resolve(REPO_ROOT, 'packages', 'benchmark', 'ref', 'metadata', 'ocr_manual.json');
const TMP = resolve(REPO_ROOT, 'packages', 'tmp');

const timingFiles = [
	'timing_0_5400.json',
	'timing_5250_7100.json',
	'timing_7100_15500.json',
	'timing_18680_35000.json',
	'timing_35000_55000.json',
	'timing_55000_80000.json',
	'timing_80000_95000.json',
	'timing_95000_110000.json',
	'timing_110000_125000.json',
	'timing_125000_148000.json',
	'timing_148000_170063.json',
];

interface Segment {
	text: string;
	start: number;
	end: number;
	frames: number;
	confidence_avg: number;
}

interface TimingOutput {
	params: any;
	segments: Segment[];
}

function loadGT() {
	const d = JSON.parse(readFileSync(GT_PATH, 'utf-8'));
	return d.result.segments as any[];
}

function isSubstringNoise(seg: Segment, others: Segment[]): boolean {
	// A segment is noise if it's entirely contained within another segment's time range
	// and its text is a substring of that segment's text
	for (const o of others) {
		if (o === seg) continue;
		if (seg.start >= o.start && seg.end <= o.end && o.text.includes(seg.text) && o.frames > seg.frames * 2) {
			return true;
		}
	}
	return false;
}

function mergeSegments(all: Segment[]): Segment[] {
	// Sort by time
	const sorted = [...all].sort((a, b) => a.start - b.start);

	// Filter out noise
	const preFilter: Segment[] = [];
	for (const s of sorted) {
		const isNoise =
			(s.text.length <= 1 && s.frames < 10) || // single-char noise like `身`, `猪`
			s.text.includes('于嘛') || // hallucination variant
			s.text === '公' ||
			s.text === '猪';
		if (isNoise) continue;
		preFilter.push(s);
	}

	// Remove substring noise: segments entirely within another segment with same text
	const filtered = preFilter.filter(s => !isSubstringNoise(s, preFilter));

	// Merge only exact same-text segments with short gaps
	const merged: Segment[] = [];
	for (const s of filtered) {
		const last = merged[merged.length - 1];
		if (last && last.text === s.text && s.start - last.end < 200) {
			last.end = Math.max(last.end, s.end);
			last.frames += s.frames;
			last.confidence_avg = (last.confidence_avg + s.confidence_avg) / 2;
		} else {
			merged.push({ ...s });
		}
	}

	// Handle boundaries: if a short segment (<100ms, <10 frames) at range boundary
	// is same text as previous and gap < 300ms (true range boundary artifact), merge
	const final: Segment[] = [];
	for (const s of merged) {
		const last = final[final.length - 1];
		if (last && last.text === s.text && s.end - s.start < 100 && s.frames < 10 && s.start - last.end < 300) {
			last.end = s.end;
			last.frames += s.frames;
		} else if (last && last.text === s.text && s.end - s.start < 100 && s.frames < 10) {
			// too far away → discard as noise
			continue;
		} else {
			final.push({ ...s });
		}
	}

	return final;
}

function main() {
	// Load all timing results
	const allSegments: Segment[] = [];
	for (const fn of timingFiles) {
		const fp = join(TMP, fn);
		if (!existsSync(fp)) {
			console.error(`Missing: ${fp}`);
			continue;
		}
		const d: TimingOutput = JSON.parse(readFileSync(fp, 'utf-8'));
		for (const s of d.segments) {
			// Ensure no zero-width segments
			if (s.end - s.start < 30) continue;
			allSegments.push(s);
		}
	}

	console.log(`Loaded ${allSegments.length} segments from all ranges`);

	const merged = mergeSegments(allSegments);
	console.log(`Merged to ${merged.length} segments`);
	console.log();

	console.log('=== All OCR segments ===');
	for (const s of merged) {
		const dur = s.end - s.start;
		console.log(`  [${s.start}-${s.end}] (${dur}ms, ${s.frames}f, conf=${s.confidence_avg.toFixed(2)}) ${s.text}`);
	}

	// Write updated GT
	const gt = JSON.parse(readFileSync(GT_PATH, 'utf-8'));
	gt.result.segments = merged.map(s => ({
		text: s.text,
		start: s.start,
		end: s.end,
		words: [],
	}));
	gt.result.text = merged.map(s => s.text).join(' ');

	writeFileSync(GT_PATH, JSON.stringify(gt, null, 2));
	console.log(`\nUpdated: ${GT_PATH}`);
}

main();
