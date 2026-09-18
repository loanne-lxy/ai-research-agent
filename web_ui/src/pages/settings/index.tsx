import { LogOut, Settings, UserPlus, Wrench } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { clearAuth, getToken } from '@/api/client.ts';
import { healthApi } from '@/api';
import { authApi, bootstrapApi } from '@/api/research';
import type { BootstrapResponse } from '@/api/research';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Spinner } from '@/components/ui/spinner.tsx';

/** Static tool list (research agent toolkit — tools/knowledge.py). Kept in
 *  the frontend: adding a tool requires a backend change anyway, so a
 *  /tools endpoint would only restate constants. */
const TOOLS = [
	{ name: 'search_events', desc: '按关键词/时间范围检索事件语料' },
	{ name: 'get_event', desc: '读取单条事件详情' },
	{ name: 'search_articles', desc: '按关键词/时间范围检索文章语料' },
	{ name: 'get_sources', desc: '列出信息源（活跃状态）' },
	{ name: 'corpus_stats', desc: '知识库规模统计（事件/文章/周次）' },
];

export function SettingsPage() {
	const navigate = useNavigate();
	const [setup, setSetup] = useState<BootstrapResponse | null>(null);
	const username = localStorage.getItem('username') ?? '';
	const [serverUrl, setServerUrl] = useState(() => localStorage.getItem('server_url') ?? '');
	const [savingServer, setSavingServer] = useState(false);
	const [serverMsg, setServerMsg] = useState<string | null>(null);
	const model = setup?.model;
	const [newUser, setNewUser] = useState('');
	const [newPw, setNewPw] = useState('');
	const [newUserMsg, setNewUserMsg] = useState<string | null>(null);

	useEffect(() => {
		let alive = true;
		bootstrapApi
			.run()
			.then((b) => alive && setSetup(b))
			.catch(() => {});
		return () => {
			alive = false;
		};
	}, []);

	const addUser = async () => {
		setNewUserMsg(null);
		try {
			await authApi.createUser(newUser.trim(), newPw);
			setNewUserMsg(`✓ ${newUser.trim()} 已创建，可登录`);
			setNewUser('');
			setNewPw('');
		} catch (e) {
			setNewUserMsg(e instanceof Error ? e.message : '创建失败');
		}
	};

	const logout = () => {
		clearAuth();
		window.location.href = '/login';
	};

	const saveServer = async () => {
		const trimmed = serverUrl.trim().replace(/\/+$/, '');
		if (!trimmed) {
			setServerMsg('地址不能为空');
			return;
		}
		setSavingServer(true);
		setServerMsg(null);
		try {
			// Same gate as setup: persist only after the address answers /health.
			await healthApi.check(trimmed, username);
			localStorage.setItem('server_url', trimmed);
			clearAuth();
			setServerMsg(`✓ 已保存，即将跳转到登录页`);
			toast.success('Server 地址已更新');
			window.setTimeout(() => (window.location.href = '/login'), 800);
		} catch (e) {
			setServerMsg(`保存失败：${e instanceof Error ? e.message : '地址不可达'}`);
		} finally {
			setSavingServer(false);
		}
	};

	return (
		<div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-6">
			<div>
				<h1 className="flex items-center gap-2 text-lg font-semibold">
					<Settings className="size-5" />
					Settings
				</h1>
				<p className="text-sm text-muted-foreground">
					模型、工具与账户（模型/工具当前为服务端固定配置）
				</p>
			</div>

			<Card>
				<CardHeader>
					<CardTitle className="text-base">Model</CardTitle>
					<CardDescription>
						研究会话绑定的模型（每个用户独立 credential，登录时自动创建）
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-2 text-sm">
					{setup ? (
						<>
							<div className="flex items-center justify-between rounded-lg border p-3">
								<div>
									<p className="font-medium">{model?.model}</p>
									<p className="text-xs text-muted-foreground">
										{model?.credential_id}
									</p>
								</div>
								<span className="text-xs text-muted-foreground">
									{model?.type}
								</span>
							</div>
							{model?.parameters && (
								<p className="text-xs text-muted-foreground">
									parameters: {JSON.stringify(model.parameters)}
								</p>
							)}
						</>
					) : (
						<div className="flex items-center gap-2 py-2">
							<Spinner className="size-4" />
							<span className="text-muted-foreground">加载模型配置…</span>
						</div>
					)}
					<Button variant="outline" size="sm" className="self-start" onClick={() => navigate('/credential')}>
						管理 credential →
					</Button>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2 text-base">
						<Wrench className="size-4" />
						Tools
					</CardTitle>
					<CardDescription>
						Research Agent 注册的工具（知识库检索，服务端固定）
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-1.5">
					{TOOLS.map((t) => (
						<div key={t.name} className="flex items-center justify-between gap-3 rounded-lg border p-2.5 text-sm">
							<span className="font-mono text-xs">{t.name}</span>
							<span className="truncate text-xs text-muted-foreground">{t.desc}</span>
						</div>
					))}
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle className="text-base">Account</CardTitle>
					<CardDescription>当前登录身份与后端连接</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-3">
					<div className="flex items-center gap-2">
						<label className="w-24 shrink-0 text-sm text-muted-foreground">Username</label>
						<Input readOnly value={username} className="max-w-xs" />
					</div>
					<div className="flex items-center gap-2">
										<label className="w-24 shrink-0 text-sm text-muted-foreground">Server</label>
										<Input
											value={serverUrl}
											onChange={(e) => setServerUrl(e.target.value)}
											placeholder="http://localhost:8000"
											className="max-w-xs font-mono"
										/>
										<Button variant="outline" size="sm" onClick={saveServer} disabled={savingServer}>
											{savingServer ? '验证中…' : '保存'}
										</Button>
										<span className="text-xs text-muted-foreground">（WSL 重启后 IP 会变，填 http://localhost:8000 最稳）</span>
									</div>
									{serverMsg && (
										<p className={`text-xs ${serverMsg.startsWith('✓') ? 'text-green-600' : 'text-red-500'}`}>
											{serverMsg}
										</p>
									)}
					<div className="flex items-center gap-2">
						<label className="w-24 shrink-0 text-sm text-muted-foreground">New user</label>
						<Input
							placeholder="username（4 位以上密码）"
							value={newUser}
							onChange={(e) => setNewUser(e.target.value)}
							className="max-w-xs"
						/>
						<Input
							placeholder="password"
							type="password"
							value={newPw}
							onChange={(e) => setNewPw(e.target.value)}
							className="max-w-[10rem]"
						/>
					</div>
					<div className="flex items-center gap-2">
						<label className="w-24 shrink-0 text-sm text-muted-foreground">Token</label>
						<p className="font-mono text-xs text-muted-foreground">
							{getToken() ? '已保存（localStorage）' : '无'}
						</p>
					</div>
					<Button variant="outline" size="sm" className="self-start" onClick={addUser}>
						<UserPlus className="size-3.5" />
						创建用户
					</Button>
					{newUserMsg && (
						<p className={`text-xs ${newUserMsg.startsWith('✓') ? 'text-green-600' : 'text-red-500'}`}>
							{newUserMsg}
						</p>
					)}
					<Button variant="destructive" size="sm" className="self-start" onClick={logout}>
						<LogOut className="size-3.5" />
						登出
					</Button>
				</CardContent>
			</Card>
		</div>
	);
}