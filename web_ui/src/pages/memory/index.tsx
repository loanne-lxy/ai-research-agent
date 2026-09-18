import { Brain, FileText, X } from 'lucide-react';
import { useEffect, useState } from 'react';

import { memoryApi } from '@/api/research';
import type { MemoryEntry } from '@/api/research';
import { Badge } from '@/components/ui/badge.tsx';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Markdown } from '@/components/markdown';
import { Spinner } from '@/components/ui/spinner.tsx';
import { toast } from 'sonner';

interface Selected {
	path: string;
	body: string;
}

function stripFrontmatter(md: string): string {
	// Memory notes carry a YAML frontmatter block; the body is what we render.
	const m = md.match(/^---\n[\s\S]*?\n---\n?/);
	return m ? md.slice(m[0].length) : md;
}

function nameOf(path: string): string {
	return path.split('/').pop()?.replace(/\.md$/, '') ?? path;
}

export function MemoryPage() {
	const [daily, setDaily] = useState<MemoryEntry[]>([]);
	const [topics, setTopics] = useState<MemoryEntry[]>([]);
	const [selected, setSelected] = useState<Selected | null>(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		let alive = true;
		memoryApi
			.list()
			.then((r) => {
				if (!alive) return;
				setDaily(r.daily);
				setTopics(r.topics);
			})
			.catch((e) => toast.error(e instanceof Error ? e.message : '加载失败'))
			.finally(() => alive && setLoading(false));
		return () => {
			alive = false;
		};
	}, []);

	const open = (f: MemoryEntry) => {
		memoryApi
			.readFile(f.path)
			.then((r) =>
				setSelected({
					path: f.path,
					body: stripFrontmatter(r.content),
				}),
			)
			.catch((e) => toast.error(e instanceof Error ? e.message : '打开失败'));
	};

	return (
		<div className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-6">
			<div>
				<h1 className="text-lg font-semibold">Memory</h1>
				<p className="text-sm text-muted-foreground">
					Agent 的长程记忆：研究偏好与从历史会话中提炼的经验
				</p>
			</div>
			{loading ? (
				<div className="flex h-40 items-center justify-center">
					<Spinner className="size-6" />
				</div>
			) : (
				<div className="grid gap-4 lg:grid-cols-3">
					<div className="flex flex-col gap-4">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2 text-base">
									<FileText className="size-4" />
									Research preferences
								</CardTitle>
								<CardDescription>
									按日沉淀的研究笔记（每日索引）
								</CardDescription>
							</CardHeader>
							<CardContent className="flex flex-col gap-1.5">
								{daily.length === 0 && (
									<p className="text-sm text-muted-foreground">空</p>
								)}
								{daily.map((f) => (
									<button
										key={f.path}
										onClick={() => open(f)}
										className="flex items-center justify-between gap-2 rounded-lg border p-2.5 text-left text-sm transition-colors hover:bg-muted/50"
									>
										<span className="truncate font-medium">{nameOf(f.path)}</span>
										<Badge variant="secondary" className="text-[10px] shrink-0">
											{(f.size / 1024).toFixed(1)} KB
										</Badge>
									</button>
								))}
							</CardContent>
						</Card>
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2 text-base">
									<Brain className="size-4" />
									Learned experience
								</CardTitle>
								<CardDescription>
									从研究会话中提炼的主题记忆
								</CardDescription>
							</CardHeader>
							<CardContent className="flex max-h-96 flex-col gap-1.5 overflow-y-auto">
								{topics.length === 0 && (
									<p className="text-sm text-muted-foreground">空</p>
								)}
								{topics.map((f) => (
									<button
										key={f.path}
										onClick={() => open(f)}
										className="flex items-center justify-between gap-2 rounded-lg border p-2.5 text-left text-sm transition-colors hover:bg-muted/50"
									>
										<span className="truncate font-medium">{nameOf(f.path)}</span>
										<Badge variant="secondary" className="text-[10px] shrink-0">
											{(f.size / 1024).toFixed(1)} KB
										</Badge>
									</button>
								))}
							</CardContent>
						</Card>
					</div>

					<Card className="lg:col-span-2 self-start">
						<CardHeader className="flex-row items-center justify-between">
							<div>
								<CardTitle className="text-base">
									{selected ? nameOf(selected.path) : '记忆内容'}
								</CardTitle>
								{selected && (
									<CardDescription>{selected.path}</CardDescription>
								)}
							</div>
							{selected && (
								<Button variant="ghost" size="icon" onClick={() => setSelected(null)}>
									<X />
								</Button>
							)}
						</CardHeader>
						<CardContent className="max-h-[70vh] overflow-y-auto">
							{selected ? (
								<Markdown>{selected.body}</Markdown>
							) : (
								<p className="py-6 text-center text-sm text-muted-foreground">
									点击左侧条目查看内容
								</p>
							)}
						</CardContent>
					</Card>
				</div>
			)}
		</div>
	);
}