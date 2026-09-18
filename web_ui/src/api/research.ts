import { client, setToken, clearAuth } from './client';

// ─── Auth ─────────────────────────────────────────────────────────────────

export interface LoginResponse {
	token: string;
	username: string;
}

export const authApi = {
	/** Login + persist the token (localStorage). Throws ApiError on bad credentials. */
	login: async (username: string, password: string) => {
		const res = await client.post<LoginResponse>(
			'/api/auth/login', { username, password }, undefined, { silent: true });
		setToken(res.token);
		return res;
	},
	/** Provision a colleague's account (any authenticated user, LAN-internal). */
	createUser: (username: string, password: string) =>
		client.post<{ username: string; created: true }>('/api/users', { username, password }),
	/** True when the stored token is still valid. */
	whoami: async () => {
		try {
			await client.get('/api/research/sessions', undefined, { silent: true });
			return true;
		} catch {
			return false;
		}
	},
	logout: () => {
		clearAuth();
		window.location.href = '/login';
	},
};

// ─── Bootstrap (per-user research setup) ──────────────────────────────────

export interface ModelBinding {
	type: string;
	credential_id: string;
	model: string;
	parameters: Record<string, number>;
}

export interface BootstrapResponse {
	credential: string;
	agent: string;
	session: string;
	model: ModelBinding;
}

export const bootstrapApi = {
	/** Idempotently ensure this user's credential + agent + session exist. */
	run: () => client.post<BootstrapResponse>('/api/bootstrap', {}),
};

// ─── Research (sidecars written by the research loop) ────────────────────

export interface ResearchSummary {
	session_id: string;
	query: string;
	verdict: string | null;
	iterations: number;
	citations: { verified: number; total: number; missing: string[] };
	updated_at: number;
}

export interface ResearchClaim {
	claim: string;
	evidence: string[];
}

export interface ResearchDetail {
	query: string;
	intent: string;
	research_plan: string[];
	evidence: Array<{ name: string; events: number; articles: number; sufficient: boolean; conflict: boolean; note: string }>;
	gaps: string[];
	conflicts: string[];
	followup_queries: string[];
	research_iterations: number;
	verdict: string | null;
	claims?: ResearchClaim[];
	citations: { verified: number; total: number; missing: string[] };
}

export const researchApi = {
	list: () => client.get<{ sessions: ResearchSummary[] }>('/api/research/sessions'),
	// silent: 404 = research never/ not yet finished here — an expected
	// state the UI renders as empty sections, not an error to toast.
	get: (sessionId: string) =>
		client.get<ResearchDetail>(`/api/research/sessions/${sessionId}`, undefined, { silent: true }),
};

// ─── Memory (ReMe) ────────────────────────────────────────────────────────

export interface MemoryEntry {
	path: string;
	size: number;
}

export const memoryApi = {
	list: () => client.get<{ daily: MemoryEntry[]; topics: MemoryEntry[] }>('/api/memory'),
	readFile: (path: string) =>
		client.post<{ path: string; content: string }>('/api/memory/file', { path }),
};

// ─── Citations (corpus resolve) ───────────────────────────────────────────

export interface CitedObject {
	type: 'event' | 'article';
	object: {
		id: string;
		title: string;
		summary?: string;
		url?: string;
		week?: string;
		category?: string;
		source?: string;
		published?: string;
	};
}

export const citationApi = {
	resolve: (ids: string[]) =>
		client.post<{ resolved: Record<string, CitedObject | null> }>('/api/citations', { ids }),
};

// ─── Knowledge base stats ─────────────────────────────────────────────────

export interface KnowledgeStats {
	articles: number;
	events: number;
	sources: number;
	sources_active: number;
	weeks: string[];
	categories: string[];
}

export const knowledgeApi = {
	stats: () => client.get<KnowledgeStats>('/api/knowledge/stats'),
};