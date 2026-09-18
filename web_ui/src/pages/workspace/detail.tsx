import {
	ArrowLeft,
	CheckCircle2,
	CircleAlert,
	Loader2,
	ListChecks,
	MessageSquareText,
	Quote,
	ScrollText,
	Send,
	Table2,
	Target,
	XCircle,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';

import type { Msg } from '@/api';
import {
	citationApi,
	researchApi,
} from '@/api/research';
import type {
	CitedObject,
	ResearchDetail,
} from '@/api/research';
import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge.tsx';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useMessages } from '@/hooks/useMessages';
import { cn } from '@/lib/utils';

/** Pull readable text out of a Msg's content blocks. */
function msgText(msg: Msg): string {
	if (typeof msg.content === 'string') return msg.content;
	return msg.content
		.filter((b) => b.type === 'text')
		.map((b) => (b as { text: string }).text)
		.join('\n');
}

export function WorkspaceDetailPage() {
	const { agentId = '', sessionId = '' } = useParams();
	const navigate = useNavigate();
	const location = useLocation();
	const prefill = (location.state as { query?: string } | null)?.query ?? '';

	const [question, setQuestion] = useState(prefill);
	const [detail, setDetail] = useState<ResearchDetail | null>(null);
	const [cited, setCited] = useState<Record<string, CitedObject | null>>({});
	const { msgs, phase, send, abort } = useMessages(agentId, sessionId);

	// Live progress + final report come from the session messages. The
	// sidecar gets milestone snapshots during the run (plan / evaluation /
	// citation) plus a final write, so refetch whenever streaming stops or
	// the phase flips between rounds.
	useEffect(() => {
		let alive = true;
		const load = () =>
			researchApi
				.get(sessionId)
				.then((d) => alive && setDetail(d))
				.catch(() => alive && setDetail(null)); // 404 = never ran here
		load();
		return () => {
			alive = false;
		};
	}, [sessionId, phase]);

	// Resolve every citation id referenced by claims into corpus objects.
	useEffect(() => {
		const ids = new Set<string>();
		detail?.claims?.forEach((c) => c.evidence.forEach((e) => ids.add(e)));
		if (ids.size === 0) return;
		let alive = true;
		citationApi
			.resolve([...ids])
			.then((r) => alive && setCited(r.resolved))
			.catch(() => {});
		return () => {
			alive = false;
		};
	}, [detail]);

	const running = phase === 'streaming';
	// Synchronous double-send guard: `phase` only flips to 'streaming'
	// once the first SSE event lands, so `running` alone lets a fast
	// double-click fire POST /chat/ twice (second one 409s). The ref is
	// set synchronously and released when phase leaves 'streaming'.
	const sendRef = useRef(false);
	useEffect(() => {
		if (phase !== 'streaming') sendRef.current = false;
	}, [phase]);

	// Accumulated report: round-1 report stays, every follow-up answer is
	// appended below with its question, so the report grows across turns.
	const reportParts = useMemo(() => {
		const parts: { question: string; answer: Msg }[] = [];
		let pendingQuestion = '';
		for (const m of msgs) {
			if (m.role === 'user') {
				pendingQuestion = msgText(m).trim();
			} else if (m.role === 'assistant' && msgText(m).trim()) {
				parts.push({ question: pendingQuestion, answer: m });
				pendingQuestion = '';
			}
		}
		return parts;
	}, [msgs]);

	// Default to the newest turn; follow the latest tab as turns accumulate.
	const [activePart, setActivePart] = useState(0);
	useEffect(() => {
		if (reportParts.length) setActivePart(reportParts.length - 1);
	}, [reportParts.length]);

	const start = () => {
		if (!question.trim() || running || sendRef.current) return;
		sendRef.current = true;
		void send([{ type: 'text', text: question.trim() } as Msg['content'][number]]);
	};

	return (
		<div className="mx-auto flex w-full max-w-6xl flex-col gap-4 p-6">
			<div className="flex items-center gap-3">
				<Button variant="ghost" size="icon" onClick={() => navigate('/')}>
					<ArrowLeft />
				</Button>
				<h1 className="text-lg font-semibold">Research Workspace</h1>
				{running && (
					<Badge variant="secondary" className="gap-1">
						<Loader2 className="size-3 animate-spin" />
						running
					</Badge>
				)}
				{detail?.verdict && !running && (
					<Badge
						variant={detail.verdict === 'sufficient' ? 'default' : 'secondary'}
					>
						{detail.verdict}
					</Badge>
				)}
				<div className="ml-auto">
					{running && (
						<Button variant="outline" size="sm" onClick={abort}>
							停止
						</Button>
					)}
				</div>
			</div>

			<div className="grid gap-4 lg:grid-cols-3">
				{/* ── left 2/3 ─────────────────────────────────────── */}
				<div className="flex flex-col gap-4 lg:col-span-2">
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<Target className="size-4" />
								Question
							</CardTitle>
						</CardHeader>
						<CardContent className="flex flex-col gap-2">
							<Textarea
								rows={2}
								value={question}
								disabled={running}
								onChange={(e) => setQuestion(e.target.value)}
								placeholder={msgs.length ? '追问…' : '研究问题…'}
							/>
							<Button onClick={start} disabled={running || !question.trim()}>
								{running ? <Loader2 className="animate-spin" /> : <Send />}
								{running ? '研究中…' : msgs.length ? '追问' : '开始研究'}
							</Button>
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<MessageSquareText className="size-4" />
								Research Progress
							</CardTitle>
							<CardDescription>
								Agent 的实时过程（规划、检索、评估、修订）
							</CardDescription>
						</CardHeader>
						<CardContent className="flex max-h-96 flex-col gap-3 overflow-y-auto">
							{msgs.length === 0 && (
								<p className="py-4 text-center text-sm text-muted-foreground">
									还没有运行记录。
								</p>
							)}
							{msgs.map((m) => (
								<div
									key={m.id}
									className={cn(
										'rounded-lg border p-3 text-sm',
										m.role === 'user' ? 'bg-muted/40' : '',
									)}
								>
									<p className="mb-1 text-xs font-medium text-muted-foreground">
										{m.role}
									</p>
									<Markdown>{msgText(m)}</Markdown>
								</div>
							))}
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<ScrollText className="size-4" />
								Final Report
							</CardTitle>
							<CardDescription>
								首轮报告 + 追问补充，逐轮累积（含引用 [evt_xxx] / [art_xxx]）
							</CardDescription>
						</CardHeader>
						<CardContent>
							{reportParts.length ? (
								<Tabs value={String(activePart)} onValueChange={(v) => setActivePart(Number(v))}>
									<TabsList className="flex-wrap">
										{reportParts.map((p, i) => (
											<TabsTrigger key={p.answer.id} value={String(i)}>
												{i === 0 ? '首轮报告' : `追问 ${i}`}
											</TabsTrigger>
										))}
									</TabsList>
									{reportParts.map((p, i) => (
										<TabsContent key={p.answer.id} value={String(i)}>
											{p.question && (
												<p className="mb-2 rounded-md bg-muted/50 p-2 text-xs text-muted-foreground">
													{p.question}
												</p>
											)}
											<Markdown>{msgText(p.answer)}</Markdown>
										</TabsContent>
									))}
								</Tabs>
							) : (
								<p className="py-4 text-center text-sm text-muted-foreground">
									完成研究后在这里显示最终报告。
								</p>
							)}
						</CardContent>
					</Card>
				</div>

				{/* ── right 1/3 ────────────────────────────────────── */}
				<div className="flex flex-col gap-4">
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<ListChecks className="size-4" />
								Plan
							</CardTitle>
						</CardHeader>
						<CardContent>
							{detail?.research_plan?.length ? (
								<ol className="flex flex-col gap-2 text-sm">
									{detail.research_plan.map((step, i) => (
										<li key={i} className="flex gap-2">
											<span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">
												{i + 1}
											</span>
											<span>{step}</span>
										</li>
									))}
								</ol>
							) : (
								<p className="text-sm text-muted-foreground">
									运行后显示研究计划。
								</p>
							)}
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<Table2 className="size-4" />
								Evidence
							</CardTitle>
							<CardDescription>
								每个维度的证据量（事件/文章）与充分性
							</CardDescription>
						</CardHeader>
						<CardContent className="flex flex-col gap-2">
							{detail?.evidence?.length ? (
								detail.evidence.map((ev, i) => (
									<div key={i} className="rounded-lg border p-2.5 text-sm">
										<div className="flex items-center justify-between gap-2">
											<span className="font-medium">{ev.name}</span>
											{ev.sufficient ? (
												<CheckCircle2 className="size-4 text-emerald-500" />
											) : (
												<CircleAlert className="size-4 text-amber-500" />
											)}
											{ev.conflict && (
												<XCircle className="size-4 text-red-500" />
											)}
										</div>
										<p className="mt-1 text-xs text-muted-foreground">
											{ev.events} 事件 · {ev.articles} 文章
											{ev.note ? ` — ${ev.note}` : ''}
										</p>
									</div>
								))
							) : (
								<p className="text-sm text-muted-foreground">
									运行后显示证据评估。
								</p>
							)}
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<Quote className="size-4" />
								Sources
							</CardTitle>
							<CardDescription>
								claims 引用的语料对象（来自本地知识库）
							</CardDescription>
						</CardHeader>
						<CardContent className="flex max-h-72 flex-col gap-2 overflow-y-auto text-sm">
							{Object.values(cited).filter(Boolean).length === 0 && (
								<p className="text-sm text-muted-foreground">
									运行后显示被引用的来源。
								</p>
							)}
							{Object.entries(cited)
								.filter(([, v]) => v)
								.map(([id, v]) => {
									const o = v!.object;
									return (
										<div key={id} className="rounded-lg border p-2.5">
											<div className="flex items-center justify-between gap-2">
												<Badge variant="secondary" className="text-[10px]">
													{v!.type}
												</Badge>
												<span className="font-mono text-[10px] text-muted-foreground">
													{id}
												</span>
											</div>
											<p className="mt-1 text-sm font-medium">{o.title}</p>
											{o.summary && (
												<p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
													{o.summary}
												</p>
											)}
											{o.url && (
												<a
													href={o.url}
													target="_blank"
													rel="noreferrer"
													className="mt-1 block truncate text-xs text-primary underline"
												>
													{o.url}
												</a>
											)}
										</div>
									);
								})}
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<Quote className="size-4" />
								Citation
							</CardTitle>
							<CardDescription>
						claim → 证据 id 的映射（引用验证状态：
						{detail?.citations
							? `${detail.citations.verified}/${detail.citations.total}`
							: '—'}
						）
					</CardDescription>
						</CardHeader>
						<CardContent className="flex flex-col gap-2 text-sm">
							{detail?.claims?.length ? (
								detail.claims.map((c, i) => (
									<div key={i} className="rounded-lg border p-2.5">
										<p>{c.claim}</p>
										<div className="mt-1.5 flex flex-wrap gap-1">
											{c.evidence.map((id) => (
												<span
													key={id}
													className={cn(
														'rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]',
														cited[id] === null && 'line-through opacity-50',
													)}
												>
													{id}
												</span>
											))}
										</div>
									</div>
								))
							) : (
								<p className="text-sm text-muted-foreground">运行后显示。</p>
							)}
						</CardContent>
					</Card>
				</div>
			</div>
		</div>
	);
}