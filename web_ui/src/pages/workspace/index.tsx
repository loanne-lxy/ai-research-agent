import {
	BookOpen,
	Brain,
	FileSearch,
	FlaskConical,
	Loader2,
	Plus,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import {
	bootstrapApi,
	knowledgeApi,
	memoryApi,
	researchApi,
} from '@/api/research';
import type {
	BootstrapResponse,
	KnowledgeStats,
	ResearchSummary,
} from '@/api/research';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Spinner } from '@/components/ui/spinner.tsx';
import { Textarea } from '@/components/ui/textarea.tsx';
import { sessionApi } from '@/api';
import { toast } from 'sonner';

function VerdictBadge({ verdict, iterations }: { verdict: string | null; iterations: number }) {
	if (!verdict) {
		return <span className="text-xs text-muted-foreground">{iterations} 轮 · 未完成</span>;
	}
	const map: Record<string, string> = {
		sufficient: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
		needs_work: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
	};
	return (
		<span className={`rounded-full px-2 py-0.5 text-xs font-medium ${map[verdict] ?? 'bg-muted text-muted-foreground'}`}>
			{verdict}
		</span>
	);
}

export function HomePage() {
	const navigate = useNavigate();
	const [setup, setSetup] = useState<BootstrapResponse | null>(null);
	const [recent, setRecent] = useState<ResearchSummary[]>([]);
	const [stats, setStats] = useState<KnowledgeStats | null>(null);
	// {daily: n, topics: n}
	const [memoryCount, setMemoryCount] = useState<{ daily: number; topics: number } | null>(null);
	const [query, setQuery] = useState('');
	const [starting, setStarting] = useState(false);

	useEffect(() => {
		let alive = true;
		(async () => {
			try {
				const [b, list, ks, mem] = await Promise.all([
					bootstrapApi.run(),
					researchApi.list(),
					knowledgeApi.stats(),
					memoryApi.list(),
				]);
				if (!alive) return;
				setSetup(b);
				setRecent(list.sessions);
				setStats(ks);
				setMemoryCount({ daily: mem.daily.length, topics: mem.topics.length });
			} catch (e) {
				toast.error(e instanceof Error ? e.message : '加载失败');
			}
		})();
		return () => {
			alive = false;
		};
	}, []);

	/** New Research: create a fresh session (bound to the Qwen credential),
	 *  then open the Workspace detail view with the question pre-filled. */
	const start = async () => {
		if (!setup) return;
		setStarting(true);
		try {
			const created = await sessionApi.create({
				agent_id: setup.agent,
				chat_model_config: setup.model,
				name: 'research',
			});
			navigate(`/workspace/${setup.agent}/${created.session_id}`, {
				state: { query: query.trim() },
			});
		} catch (e) {
			toast.error(e instanceof Error ? e.message : '创建研究会话失败');
		} finally {
			setStarting(false);
		}
	};

	if (!setup) {
		return (
			<div className="flex h-full items-center justify-center">
				<Spinner className="size-6" />
			</div>
		);
	}

	return (
		<div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
			{/* New Research */}
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2">
						<FlaskConical className="size-4" />
						New Research
					</CardTitle>
					<CardDescription>
						输入研究问题，Agent 会基于知识库（{stats?.events ?? '…'} 个事件 /{' '}
						{stats?.articles ?? '…'} 篇文章）检索证据并生成报告
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-3">
					<Textarea
						rows={3}
						placeholder="例如：2026 年上半年多智能体系统的记忆与个性化方向有哪些进展？"
						value={query}
						onChange={(e) => setQuery(e.target.value)}
					/>
					<Button onClick={start} disabled={starting || !query.trim()}>
						{starting ? <Loader2 className="animate-spin" /> : <Plus />}
						开始研究
					</Button>
				</CardContent>
			</Card>

			<div className="grid gap-6 lg:grid-cols-3">
				{/* Recent Research */}
				<Card className="lg:col-span-2">
					<CardHeader className="flex-row items-center justify-between">
						<div>
							<CardTitle className="flex items-center gap-2 text-base">
								<FileSearch className="size-4" />
								Recent Research
							</CardTitle>
							<CardDescription>最近的研究任务（含 CLI 运行）</CardDescription>
						</div>
						<Button variant="ghost" size="sm" onClick={() => navigate('/history')}>
							全部 →
						</Button>
					</CardHeader>
					<CardContent className="flex flex-col gap-2">
						{recent.length === 0 && (
							<p className="py-6 text-center text-sm text-muted-foreground">还没有研究记录，从上面开始一个。</p>
						)}
						{recent.slice(0, 5).map((r) => (
							<button
								key={r.session_id}
								onClick={() => navigate(`/workspace/${setup.agent}/${r.session_id}`)}
								className="flex items-center justify-between gap-3 rounded-lg border p-3 text-left transition-colors hover:bg-muted/50"
							>
								<div className="min-w-0">
									<p className="truncate text-sm font-medium">{r.query || '(无标题)'}</p>
									<p className="mt-0.5 text-xs text-muted-foreground">
										{new Date(r.updated_at * 1000).toLocaleString()}
										{r.citations.total > 0 && ` · 引用 ${r.citations.verified}/${r.citations.total}`}
									</p>
								</div>
								<VerdictBadge verdict={r.verdict} iterations={r.iterations} />
							</button>
						))}
					</CardContent>
				</Card>

				<div className="flex flex-col gap-6">
					{/* Knowledge Overview */}
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<BookOpen className="size-4" />
								Knowledge
							</CardTitle>
						</CardHeader>
						<CardContent className="grid grid-cols-2 gap-3 text-center">
							<div className="rounded-lg bg-muted/50 p-3">
								<p className="text-xl font-semibold">{stats?.events ?? '…'}</p>
								<p className="text-xs text-muted-foreground">事件</p>
							</div>
							<div className="rounded-lg bg-muted/50 p-3">
								<p className="text-xl font-semibold">{stats?.articles ?? '…'}</p>
								<p className="text-xs text-muted-foreground">文章</p>
							</div>
							<div className="rounded-lg bg-muted/50 p-3">
								<p className="text-xl font-semibold">{stats?.sources_active ?? '…'}</p>
								<p className="text-xs text-muted-foreground">活跃源</p>
							</div>
							<div className="rounded-lg bg-muted/50 p-3">
								<p className="text-xl font-semibold">{stats?.weeks?.length ?? '…'}</p>
								<p className="text-xs text-muted-foreground">周次</p>
							</div>
						</CardContent>
						<CardContent>
							<Button variant="outline" size="sm" className="w-full" onClick={() => navigate('/knowledge')}>
								打开知识库 →
							</Button>
						</CardContent>
					</Card>

					{/* Memory snapshot */}
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2 text-base">
								<Brain className="size-4" />
								Memory
							</CardTitle>
						</CardHeader>
						<CardContent className="flex items-center justify-between">
							<p className="text-sm text-muted-foreground">
								{memoryCount
									? `长程记忆：${memoryCount.topics} 个主题 / ${memoryCount.daily} 篇日注`
									: '加载中…'}
							</p>
							<Button variant="ghost" size="sm" onClick={() => navigate('/memory')}>
								查看 →
							</Button>
						</CardContent>
					</Card>
				</div>
			</div>
		</div>
	);
}