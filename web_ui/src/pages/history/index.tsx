import { ArrowUpRight, History, Play } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { sessionApi } from '@/api';
import { bootstrapApi, researchApi } from '@/api/research';
import type { BootstrapResponse, ResearchSummary } from '@/api/research';
import { Badge } from '@/components/ui/badge.tsx';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Spinner } from '@/components/ui/spinner.tsx';
import { toast } from 'sonner';

export function HistoryPage() {
	const navigate = useNavigate();
	const [setup, setSetup] = useState<BootstrapResponse | null>(null);
	const [list, setList] = useState<ResearchSummary[]>([]);

	useEffect(() => {
		let alive = true;
		Promise.all([bootstrapApi.run(), researchApi.list()])
			.then(([b, r]) => {
				if (!alive) return;
				setSetup(b);
				setList(r.sessions);
			})
			.catch((e) => toast.error(e instanceof Error ? e.message : '加载失败'));
		return () => {
			alive = false;
		};
	}, []);

	/** Resume: if the session still exists in storage (web-created) open it
	 *  directly; otherwise it's a CLI-only sidecar — start a fresh session
	 *  with the same question pre-filled. */
	const resume = async (r: ResearchSummary) => {
		if (!setup) return;
		try {
			await sessionApi.messages(r.session_id, setup.agent);
			navigate(`/workspace/${setup.agent}/${r.session_id}`);
		} catch {
			try {
				const created = await sessionApi.create({
					agent_id: setup.agent,
					chat_model_config: setup.model,
					name: 'research',
				});
				navigate(`/workspace/${setup.agent}/${created.session_id}`, {
					state: { query: r.query },
				});
			} catch (e) {
				toast.error(e instanceof Error ? e.message : '恢复失败');
			}
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
		<div className="mx-auto flex w-full max-w-4xl flex-col gap-4 p-6">
			<div>
				<h1 className="text-lg font-semibold">History</h1>
				<p className="text-sm text-muted-foreground">
					全部研究记录（web + CLI），可继续追问或重开
				</p>
			</div>
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2 text-base">
						<History className="size-4" />
						Previous research
					</CardTitle>
					<CardDescription>{list.length} 条记录</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{list.length === 0 && (
						<p className="py-6 text-center text-sm text-muted-foreground">
							还没有研究记录。
						</p>
					)}
					{list.map((r) => (
						<div
							key={r.session_id}
							className="flex items-center justify-between gap-3 rounded-lg border p-3"
						>
							<button
								className="min-w-0 flex-1 text-left"
								onClick={() =>
									navigate(`/workspace/${setup.agent}/${r.session_id}`)
								}
							>
								<p className="truncate text-sm font-medium">
									{r.query || '(无标题)'}
								</p>
								<p className="mt-0.5 text-xs text-muted-foreground">
									{new Date(r.updated_at * 1000).toLocaleString()} · {r.iterations} 轮
									{r.citations.total > 0 &&
										` · 引用 ${r.citations.verified}/${r.citations.total}`}
								</p>
							</button>
							{r.verdict && (
								<Badge variant="secondary">{r.verdict}</Badge>
							)}
							<Button
								variant="outline"
								size="sm"
								className="gap-1 shrink-0"
								onClick={() => resume(r)}
							>
								<Play className="size-3.5" />
								Resume
							</Button>
							<Button
								variant="ghost"
								size="icon"
								onClick={() =>
									navigate(`/workspace/${setup.agent}/${r.session_id}`)
								}
							>
								<ArrowUpRight />
							</Button>
						</div>
					))}
				</CardContent>
			</Card>
		</div>
	);
}