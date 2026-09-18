import { Loader2 } from 'lucide-react';
import { useState } from 'react';

import { authApi } from '@/api/research';
import { Button } from '@/components/ui/button.tsx';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card.tsx';
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Alert, AlertDescription } from '@/components/ui/alert.tsx';

interface Props {
	onComplete: () => void;
}

/** JWT login. On success the token is in localStorage and every subsequent
 *  request (client.ts buildHeaders) carries it as Authorization: Bearer. */
export function LoginPage({ onComplete }: Props) {
	const [username, setUsername] = useState('');
	const [password, setPassword] = useState('');
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState('');

	const submit = async (e: React.FormEvent) => {
		e.preventDefault();
		setBusy(true);
		setError('');
		try {
			await authApi.login(username.trim(), password);
			onComplete();
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Login failed');
		} finally {
			setBusy(false);
		}
	};

	return (
		<div className="flex items-center justify-center h-screen bg-muted/30">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>Research Agent</CardTitle>
					<CardDescription>Sign in to your research workspace</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={submit}>
						<FieldGroup>
							{error && (
								<Alert variant="destructive">
									<AlertDescription>{error}</AlertDescription>
								</Alert>
							)}
							<Field>
								<FieldLabel htmlFor="login-user">Username</FieldLabel>
								<Input
									id="login-user"
									value={username}
									onChange={(e) => setUsername(e.target.value)}
									autoComplete="username"
									required
								/>
							</Field>
							<Field>
								<FieldLabel htmlFor="login-pass">Password</FieldLabel>
								<Input
									id="login-pass"
									type="password"
									value={password}
									onChange={(e) => setPassword(e.target.value)}
									autoComplete="current-password"
									required
								/>
							</Field>
							<Button type="submit" disabled={busy} className="w-full">
								{busy && <Loader2 className="animate-spin" />}
								Sign in
							</Button>
						</FieldGroup>
					</form>
				</CardContent>
			</Card>
		</div>
	);
}