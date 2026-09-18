import { Navigate, Outlet } from 'react-router-dom';

import { isAuthed } from '@/api/client.ts';
import { AppSidebar } from '@/components/layout/AppSidebar';
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar';

export function AppLayout() {
	// Auth gate lives inside the router tree — <Navigate> needs router
	// context; rendering it in App() (pre-RouterProvider) crashes → blank.
	if (!isAuthed()) return <Navigate to="/login" replace />;
	return (
		<div className="h-screen flex">
			<SidebarProvider>
				<AppSidebar />
				<SidebarInset className="flex-1 overflow-y-auto bg-canvas">
					<Outlet />
				</SidebarInset>
			</SidebarProvider>
		</div>
	);
}
