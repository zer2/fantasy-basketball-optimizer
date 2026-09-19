// table/failure_message.ts
// Turns a failed backend call into a sentence, shown where the candidate table normally is.
//
// Every one of these used to reach a `console.error` and nothing else: the table kept whatever
// it was last showing, the indicator settled back to idle, and the app looked like it had
// worked. The table's own space is the right place for the news, because the table is the thing
// that is missing.
//
// The backend already writes an actionable sentence for the cases it can name — zero projection
// weights, an unusable board, a league it cannot reach — so those are shown as written rather
// than paraphrased here. What this adds is the framing for the ones whose raw text explains
// nothing ("Internal Server Error"), and a category for each so the tone can differ.

import { HTTPError, readErrorDetail } from '../api/client.js'
import { showTableMessage } from './player_table.js'

/** Why the call failed, from the caller's point of view rather than the server's. */
export type FailureKind =
    | 'projection-weights'   // the pool cannot be built from the chosen sources
    | 'unauthenticated'      // the platform (or the app) needs a sign-in first
    | 'room-unavailable'     // the league exists but could not be read right now
    | 'settings'             // some setting the user can change is rejected
    | 'unknown'              // anything we cannot explain: a real fault

export interface Failure {
    kind: FailureKind
    message: string
}

/** The sentence to show, and what kind of failure produced it. */
export function describeFailure(error: unknown): Failure {
    if (!(error instanceof HTTPError)) {
        // A network drop, a CORS refusal, a bug in our own code — nothing came back to read.
        const detail = error instanceof Error ? error.message : String(error)
        return { kind: 'unknown', message: `Something went wrong. ${detail}` }
    }

    const detail = readErrorDetail(error.body).trim()

    if (error.status === 401) {
        // The backend distinguishes "sign in" from "connect Yahoo", and says which.
        return { kind: 'unauthenticated', message: detail }
    }
    if (error.status === 502 || error.status === 503 || error.status === 504) {
        return {
            kind: 'room-unavailable',
            message: `${detail} The league may not be open yet, or the platform may be unreachable.`,
        }
    }
    if (error.status === 400 || error.status === 422) {
        // A 422 is Pydantic's, and its detail is a LIST of field errors, not a sentence —
        // readErrorDetail hands back the raw envelope in that case, which is no use on screen.
        // Its "loc" key is the reliable marker, since the envelope itself is still an object.
        const isFieldErrorList = detail.includes('"loc"')
        const readable = isFieldErrorList
            ? 'One of the league settings is out of range. Check the sidebar values.'
            : detail
        const isProjectionWeights = /blend weight/i.test(readable)
        return { kind: isProjectionWeights ? 'projection-weights' : 'settings', message: readable }
    }
    // 500s carry "Internal Server Error" or a deliberately vague sentence, neither of which tells
    // the user anything — so say plainly that it is not their fault and not their fix.
    return {
        kind: 'unknown',
        message: 'Something went wrong on the server, so the rankings could not be produced. '
               + 'Try again; if it keeps happening the details are in the server log.',
    }
}

/**
 * Show a failed call in the table's place.
 *
 * Aborted calls are not failures — a superseded evaluate aborts the one before it every time a
 * setting changes — so they never reach the screen.
 */
export function showFailureInTable(error: unknown): boolean {
    if (error instanceof Error && error.name === 'AbortError') return false
    const failure = describeFailure(error)
    showTableMessage(failure.message, { isError: true })
    return true
}
