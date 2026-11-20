import { createAuthClient } from '@neon/neon-js/auth'

export const authClient = createAuthClient(`${window.location.origin}/auth`)
