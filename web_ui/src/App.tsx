import { Onborda, OnbordaProvider } from 'onborda';
import { useMemo, useState } from 'react';
import { createBrowserRouter, Navigate, RouterProvider, useNavigate } from 'react-router-dom';
import { Toaster } from 'sonner';

import { MCPHubPage } from './pages/mcp';
import { SkillHubPage } from './pages/skill';
import { RouteError } from '@/components/error/RouteError';
import { AppLayout } from '@/components/layout/AppLayout';
import { buildChatTour } from '@/components/tour/chatTourSteps';
import { TourCard } from '@/components/tour/TourCard';
import { UploadProvider } from '@/context/UploadContext';
import { isAuthed } from '@/api/client.ts';
import { useTranslation } from '@/i18n/useI18n';
import { ChannelPage } from '@/pages/channel';
import { ChatPage } from '@/pages/chat';
import { CredentialPage } from '@/pages/credential';
import { HistoryPage } from '@/pages/history';
import { KnowledgePage } from '@/pages/knowledge';
import { LoginPage } from '@/pages/login';
import { MemoryPage } from '@/pages/memory';
import { SchedulePage } from '@/pages/schedule';
import { SetupPage } from '@/pages/setup';
import { SettingsPage } from '@/pages/settings';
import { WorkspaceDetailPage } from '@/pages/workspace/detail';
import { HomePage } from '@/pages/workspace';

function SetupPageRoute() {
	const navigate = useNavigate();
	return (
		<>
			<div className="h-screen">
				<SetupPage onComplete={() => navigate('/')} />
			</div>
			<Toaster richColors position="top-right" />
		</>
	);
}

const router = createBrowserRouter([
	{
		element: <AppLayout />,
		errorElement: <RouteError />,
		children: [
			{
				// Content-level boundary: a crash in a page replaces only
				// the Outlet area, so AppLayout (the icon rail / nav) stays
				// usable. The parent route keeps its own errorElement as a
				// last-resort catch-all for AppLayout/AppSidebar crashes.
				errorElement: <RouteError />,
				children: [
					{ path: '/', element: <HomePage /> },
					{
						path: '/workspace/:agentId/:sessionId',
						element: <WorkspaceDetailPage />,
					},
					{
						path: '/chat/:agentId?/:sessionId?/:memberId?',
						element: <ChatPage />,
					},
					{ path: '/history', element: <HistoryPage /> },
					{ path: '/memory', element: <MemoryPage /> },
					{ path: '/settings', element: <SettingsPage /> },
					{ path: '/schedule', element: <SchedulePage /> },
					{ path: '/channel', element: <ChannelPage /> },
					{ path: '/credential', element: <CredentialPage /> },
					{ path: '/mcp', element: <MCPHubPage /> },
					{ path: '/mcp/:hubId', element: <MCPHubPage /> },
					{ path: '/skill', element: <SkillHubPage /> },
					{ path: '/skill/:hubId', element: <SkillHubPage /> },
					{ path: '/knowledge', element: <KnowledgePage /> },
					{ path: '/knowledge/:kbId', element: <KnowledgePage /> },
				],
			},
		],
	},
	{
		// Login is outside AppLayout (no sidebar) — it is the auth gate.
		path: '/login',
		element: <LoginPageRouter />,
		errorElement: <RouteError />,
	},
	{ path: '/setup', element: <SetupPageRoute />, errorElement: <RouteError /> },
]);

function LoginPageRouter() {
	const navigate = useNavigate();
	if (isAuthed()) return <Navigate to="/" replace />;
	return (
		<div className="h-screen">
			<LoginPage onComplete={() => navigate('/')} />
			<Toaster richColors position="top-right" />
		</div>
	);
}

function App() {
	const { t } = useTranslation();
	const [setupComplete, setSetupComplete] = useState(() => !!localStorage.getItem('server_url'));
	const tours = useMemo(() => [buildChatTour(t)], [t]);

	if (!setupComplete) {
		return <SetupPage onComplete={() => setSetupComplete(true)} />;
	}
	// JWT gate lives in AppLayout (inside the router) — see there.

	return (
		<OnbordaProvider>
			<Onborda
				steps={tours}
				cardComponent={TourCard}
				shadowOpacity="0.6"
				cardTransition={{ type: 'spring', duration: 0.4 }}
			>
				<UploadProvider>
					<RouterProvider router={router} />
				</UploadProvider>
				<Toaster richColors position="top-right" />
			</Onborda>
		</OnbordaProvider>
	);
}

export default App;
