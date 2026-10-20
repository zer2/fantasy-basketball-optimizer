// data_entry/auction_autodraft.ts
// Auction autodrafting: nomination rotation, ascending $1 round-robin bidding with permanent dropout
// on a pass, and interactive turns for manual drafters. The backend supplies each nomination, the
// autodrafters' valuations and every team's maximum allowable bid; the bidding runs here, in one place.
//
// This module never renders. It announces every state change with AUCTION_AUTODRAFT_CHANGED on
// document, and the auction board re-renders on it.

import { getSessionId, setAutopilotOn, setAutopilotOff, withSessionRetry } from '../api/session.js'
import { nominateAuctionPlayer, valueAuctionPlayer, AuctionNominationResult, HTTPError, readErrorDetail } from '../api/client.js'
import { getCurrentSeat } from '../app_state.js'
import { getAuctionDrafterMethod } from './drafter_methods.js'
import {
    recordAuctionPick,
    getNominatorIndex,
    isAuctionComplete,
    getAuctionState,
    getTeamIdentitiesFromBoard,
    getNDrafters,
} from './auction_state.js'

export const AUCTION_AUTODRAFT_CHANGED = 'auction-autodraft-changed'

export interface PendingNomination {
    playerId: number
    playerName: string
    nominatorIndex: number
    currentBid: number
    highBidderIndex: number
    activeBidders: Set<number>               // drafters who have not passed on this player
    valuations: Record<string, number>       // team -> dollar valuation, autodrafters only
    maxAllowedBids: Record<number, number>   // drafter index -> most it may bid
    turnIndex: number                        // next drafter in order to act
    bidHistory: Array<{ team: string; bid: number; action: string }>
    lastBids: Record<number, number>         // highest bid placed by each drafter on this player
}

type BiddingOutcome = 'complete' | 'manual-turn' | 'awaiting-valuation'

// ─── Module state ─────────────────────────────────────────────────────────────

let autopilotRunning = false
let autopilotCancelRequested = false
// Aborts the autopilot's in-flight nomination request, so Stop takes effect at once rather than after the
// request's evaluates finish. The backend still completes that request; its result is discarded.
let autopilotAbortController: AbortController | null = null
// Set by Stop, or by a failure: either blocks the automatic restart that every board render would
// otherwise trigger, until the user continues.
let autopilotStopped = false
let autodraftErrorMessage: string | null = null
let pendingNomination: PendingNomination | null = null
let nominationRequestInFlight = false
const valuationRequestsInFlight = new Set<string>()
// Bumped whenever the board is reset, so a request that was in flight across the reset is discarded.
let boardGeneration = 0

export function isAutopilotRunning(): boolean            { return autopilotRunning }
export function isAutopilotStopped(): boolean            { return autopilotStopped }
export function getAutodraftErrorMessage(): string | null { return autodraftErrorMessage }
export function getPendingNomination(): PendingNomination | null { return pendingNomination }
export function isNominationRequestInFlight(): boolean   { return nominationRequestInFlight }

/** Whether the drafter is configured as an auction autodrafter. */
export function isAuctionAutodrafter(drafterIndex: number): boolean {
    return getAuctionDrafterMethod(drafterIndex) !== 'Manual input'
}

/** The board index of the user's seat, or -1 while no seat is selected. Used only to word turn prompts. */
export function getCurrentSeatIndex(): number {
    const seatName = getCurrentSeat()
    return seatName === null ? -1 : getTeamIdentitiesFromBoard().indexOf(seatName)
}

/** Whether the drafter is an autodrafter still waiting for its valuation of the player up for bid. */
export function isAwaitingValuation(drafterIndex: number): boolean {
    if (!pendingNomination || !isAuctionAutodrafter(drafterIndex)) return false
    return !(readTeamIdentity(drafterIndex) in pendingNomination.valuations)
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

function notifyAuctionAutodraftChanged(): void {
    document.dispatchEvent(new CustomEvent(AUCTION_AUTODRAFT_CHANGED))
}

/** Drops any nomination and error, cancels a running autopilot, and discards in-flight requests.
 *  Call when the board is reset, cleared, or reconfigured. */
export function resetAuctionAutodraft(): void {
    boardGeneration += 1
    pendingNomination = null
    autodraftErrorMessage = null
    autopilotStopped = false
    if (autopilotRunning) cancelAutopilotRun()
}

function cancelAutopilotRun(): void {
    autopilotCancelRequested = true
    autopilotAbortController?.abort()
}

/** Takes the player up for bid off the block without awarding him. When an autodrafter nominated him, the
 *  autopilot stops as well, so it does not put the same player straight back up. */
export function cancelPendingNomination(): void {
    if (pendingNomination && isAuctionAutodrafter(pendingNomination.nominatorIndex)) autopilotStopped = true
    pendingNomination = null
    notifyAuctionAutodraftChanged()
}

export function stopAuctionAutopilot(): void {
    autopilotStopped = true
    cancelAutopilotRun()
    notifyAuctionAutodraftChanged()
}

/** Clears a stop or an error and picks up where the auction left off: the bidding on a player up for bid
 *  (retrying any valuation that failed), or else the autopilot when an autodrafter is due to nominate. */
export function continueAuctionAutopilot(): void {
    autopilotStopped = false
    autodraftErrorMessage = null
    try {
        if (pendingNomination) {
            continueNomination(pendingNomination)
            return
        }
        notifyAuctionAutodraftChanged()
        startAuctionAutopilotIfDue()
    } catch (error) {
        reportAutodraftError(error)
    }
}

/** Starts the autopilot when an autodrafter is due to nominate and nothing holds it back. */
export function startAuctionAutopilotIfDue(): void {
    if (autopilotRunning || autopilotStopped || autodraftErrorMessage !== null) return
    if (pendingNomination || nominationRequestInFlight || isAuctionComplete()) return
    if (!isAuctionAutodrafter(getNominatorIndex())) return
    void runAuctionAutopilot()
}

function reportAutodraftError(error: unknown): void {
    console.error('Auction autodraft failed:', error)
    autodraftErrorMessage = error instanceof HTTPError
        ? readErrorDetail(error.body)
        : (error instanceof Error ? error.message : String(error))
    notifyAuctionAutodraftChanged()
}

/** The open session's id. Called inside withSessionRetry, which has just ensured one exists (and replaces
 *  an expired one, e.g. after a backend restart, on a 404). */
function requireSessionId(): string {
    const sessionId = getSessionId()
    if (!sessionId) throw new Error('No session is open, so the auction cannot reach the backend.')
    return sessionId
}

function readTeamIdentity(drafterIndex: number): string {
    const identity = getTeamIdentitiesFromBoard()[drafterIndex]
    if (identity === undefined) throw new Error(`Drafter ${drafterIndex} has no team on the board.`)
    return identity
}

function listAutodrafterTeamIds(): string[] {
    return getTeamIdentitiesFromBoard().filter((_, index) => isAuctionAutodrafter(index))
}

// ─── Bidding ──────────────────────────────────────────────────────────────────

function buildPendingNomination(
    result: AuctionNominationResult
    , nominatorIndex: number
): PendingNomination {
    const nDrafters = getNDrafters()
    const maxAllowedBids: Record<number, number> = {}
    for (let drafterIndex = 0; drafterIndex < nDrafters; drafterIndex++) {
        const team = readTeamIdentity(drafterIndex)
        if (!(team in result.max_allowed_bids)) throw new Error(`The nomination carried no maximum bid for ${team}.`)
        maxAllowedBids[drafterIndex] = result.max_allowed_bids[team]
    }
    const activeBidders = new Set<number>()
    for (let drafterIndex = 0; drafterIndex < nDrafters; drafterIndex++) {
        if (maxAllowedBids[drafterIndex] >= result.opening_bid) activeBidders.add(drafterIndex)
    }
    activeBidders.add(nominatorIndex)
    const nomination: PendingNomination = {
        playerId:        result.nominated_player_id,
        playerName:      result.nominated_player_name,
        nominatorIndex,
        currentBid:      result.opening_bid,
        highBidderIndex: nominatorIndex,
        activeBidders,
        valuations:      { ...result.valuations },
        maxAllowedBids,
        turnIndex:       (nominatorIndex + 1) % nDrafters,
        bidHistory:      [{ team: readTeamIdentity(nominatorIndex), bid: result.opening_bid, action: 'nominate' }],
        lastBids:        { [nominatorIndex]: result.opening_bid },
    }
    settleAutodrafterBidding(nomination)
    return nomination
}

/**
 * Plays out the bidding among the autodrafters alone, in the usual turn order and $1 steps, with manual
 * drafters sitting out (not passing), then hands the turn to the seat after the high bidder.
 *
 * Without this, a manual drafter is prompted on every round from the $1 opening, although no manual bid
 * below the price the autodrafters reach on their own can win: the autodrafter that values the player
 * most would outbid it. The autodrafters' result is the same as in full $1 bidding (the second-highest
 * autodrafter limit, or $1 above it, held by the highest), so manual drafters now first act at that price
 * and can bid it up from there. With no manual drafter in the nomination it decides the whole auction.
 */
function settleAutodrafterBidding(nomination: PendingNomination): void {
    const nDrafters = getNDrafters()
    let turnsWithoutRaise = 0
    while (turnsWithoutRaise < nDrafters) {
        const drafterIndex = nomination.turnIndex
        nomination.turnIndex = (drafterIndex + 1) % nDrafters
        turnsWithoutRaise += 1

        const isSettling = nomination.activeBidders.has(drafterIndex)
            && drafterIndex !== nomination.highBidderIndex
            && isAuctionAutodrafter(drafterIndex)
            // A team switched to autodraft while this nomination was being requested has no valuation yet;
            // the regular bidding fetches it when its turn comes.
            && readTeamIdentity(drafterIndex) in nomination.valuations
        if (!isSettling) continue

        const team = readTeamIdentity(drafterIndex)
        const nextBid = nomination.currentBid + 1
        if (nextBid <= nomination.valuations[team] && nextBid <= nomination.maxAllowedBids[drafterIndex]) {
            nomination.currentBid = nextBid
            nomination.highBidderIndex = drafterIndex
            nomination.lastBids[drafterIndex] = nextBid
            nomination.bidHistory.push({ team, bid: nextBid, action: 'raise' })
            turnsWithoutRaise = 0
        } else {
            nomination.activeBidders.delete(drafterIndex)
            nomination.bidHistory.push({ team, bid: nomination.currentBid, action: 'pass' })
        }
    }
    nomination.turnIndex = (nomination.highBidderIndex + 1) % nDrafters
}

/**
 * Advances the nomination through consecutive autodrafters in set order. An autodrafter raises by $1 when
 * the next bid is within both its valuation and its maximum allowable bid; otherwise it passes and drops
 * out for this player. A manual drafter whose maximum allowable bid is below the next bid has no bid to
 * make, so it passes the same way instead of being prompted. Stops when the turn returns to the high bidder
 * or one bidder is left ('complete'), when a manual drafter must act ('manual-turn'), or when an autodrafter
 * has no valuation yet because it was switched on mid-nomination ('awaiting-valuation').
 */
function advanceAutobidders(nomination: PendingNomination): BiddingOutcome {
    const nDrafters = getNDrafters()
    while (true) {
        if (nomination.activeBidders.size <= 1) return 'complete'

        const drafterIndex = nomination.turnIndex
        if (!nomination.activeBidders.has(drafterIndex)) {
            nomination.turnIndex = (drafterIndex + 1) % nDrafters
            continue
        }
        if (drafterIndex === nomination.highBidderIndex) return 'complete'

        const team = readTeamIdentity(drafterIndex)
        const nextBid = nomination.currentBid + 1
        const canAffordNextBid = nextBid <= nomination.maxAllowedBids[drafterIndex]
        if (!isAuctionAutodrafter(drafterIndex) && canAffordNextBid) return 'manual-turn'
        if (isAuctionAutodrafter(drafterIndex) && !(team in nomination.valuations)) return 'awaiting-valuation'

        const isAutodrafterRaise = isAuctionAutodrafter(drafterIndex) && nextBid <= nomination.valuations[team]
        if (canAffordNextBid && isAutodrafterRaise) {
            nomination.currentBid = nextBid
            nomination.highBidderIndex = drafterIndex
            nomination.lastBids[drafterIndex] = nextBid
            nomination.bidHistory.push({ team, bid: nextBid, action: 'raise' })
        } else {
            nomination.activeBidders.delete(drafterIndex)
            nomination.bidHistory.push({ team, bid: nomination.currentBid, action: 'pass' })
        }
        nomination.turnIndex = (drafterIndex + 1) % nDrafters
    }
}

/** Records the player for the high bidder at the current bid. */
function awardNomination(nomination: PendingNomination): void {
    const recorded = recordAuctionPick(nomination.playerId, nomination.currentBid, nomination.highBidderIndex)
    if (!recorded) {
        throw new Error(`${nomination.playerName} could not be awarded to ${readTeamIdentity(nomination.highBidderIndex)}`
                        + ': that team has no empty roster slot.')
    }
    if (pendingNomination === nomination) pendingNomination = null
}

/** Runs the bidding on from the current turn: awards the player when bidding is over (then lets the
 *  autopilot take the next nomination), or parks the nomination for a manual turn or a valuation. */
function continueNomination(nomination: PendingNomination): void {
    const outcome = advanceAutobidders(nomination)
    if (outcome === 'complete') {
        awardNomination(nomination)
        notifyAuctionAutodraftChanged()
        startAuctionAutopilotIfDue()
        return
    }
    pendingNomination = nomination
    if (outcome === 'awaiting-valuation') void fetchMissingValuations(nomination)
    notifyAuctionAutodraftChanged()
}

/** Fetches valuations for autodrafters that have none for the player up for bid, then resumes bidding. */
async function fetchMissingValuations(nomination: PendingNomination): Promise<void> {
    const missingTeams = getTeamIdentitiesFromBoard().filter((team, index) =>
        isAuctionAutodrafter(index) && !(team in nomination.valuations) && !valuationRequestsInFlight.has(team))
    if (missingTeams.length === 0) return
    missingTeams.forEach(team => valuationRequestsInFlight.add(team))
    const generation = boardGeneration
    try {
        const valuations = await withSessionRetry(() => valueAuctionPlayer(requireSessionId(), {
            ...getAuctionState(),
            player_id:          nomination.playerId,
            valuation_team_ids: missingTeams,
        }))
        if (generation !== boardGeneration || pendingNomination !== nomination) return
        Object.assign(nomination.valuations, valuations)
        continueNomination(nomination)
    } catch (error) {
        reportAutodraftError(error)
    } finally {
        missingTeams.forEach(team => valuationRequestsInFlight.delete(team))
        notifyAuctionAutodraftChanged()
    }
}

// ─── Autopilot ────────────────────────────────────────────────────────────────

/**
 * Runs consecutive autodrafter nominations until the auction is complete, a manual drafter is due to
 * nominate, a nomination waits on a manual drafter, or the user stops it. A failure stops it and is shown
 * on the board; it does not restart until the user continues.
 */
async function runAuctionAutopilot(): Promise<void> {
    if (autopilotRunning) return
    autopilotRunning = true
    autopilotCancelRequested = false
    autopilotAbortController = new AbortController()
    const signal = autopilotAbortController.signal
    const generation = boardGeneration
    setAutopilotOn()
    notifyAuctionAutodraftChanged()

    try {
        while (!isAuctionComplete() && !autopilotCancelRequested) {
            const nominatorIndex = getNominatorIndex()
            if (!isAuctionAutodrafter(nominatorIndex)) break

            const result = await withSessionRetry(() => nominateAuctionPlayer(requireSessionId(), {
                ...getAuctionState(),
                nominator_id:        readTeamIdentity(nominatorIndex),
                nominated_player_id: null,
                opening_bid:         null,
                valuation_team_ids:  listAutodrafterTeamIds(),
            }, signal))
            if (autopilotCancelRequested || generation !== boardGeneration) break
            // The nominator was switched to manual while its nomination was in flight: it no longer nominates.
            if (!isAuctionAutodrafter(nominatorIndex)) break

            const nomination = buildPendingNomination(result, nominatorIndex)
            const outcome = advanceAutobidders(nomination)
            if (outcome !== 'complete') {
                pendingNomination = nomination
                if (outcome === 'awaiting-valuation') void fetchMissingValuations(nomination)
                break
            }
            awardNomination(nomination)
            notifyAuctionAutodraftChanged()
        }
    } catch (error) {
        // An abort is Stop (or a board reset) cutting the request short, not a failure.
        if (!signal.aborted) {
            autopilotStopped = true
            reportAutodraftError(error)
        }
    } finally {
        autopilotRunning = false
        autopilotAbortController = null
        setAutopilotOff()
        notifyAuctionAutodraftChanged()
    }
}

// ─── Manual actions ───────────────────────────────────────────────────────────

/** The drafter whose turn it is raises the current bid to `bid`. */
export function submitManualBid(
    bid: number
    , drafterIndex: number
): void {
    try {
        const nomination = pendingNomination
        if (!nomination) throw new Error('There is no player up for bid.')
        if (drafterIndex !== nomination.turnIndex) throw new Error(`It is not ${readTeamIdentity(drafterIndex)}'s turn to bid.`)
        if (drafterIndex === nomination.highBidderIndex) throw new Error(`${readTeamIdentity(drafterIndex)} already holds the high bid.`)
        if (!Number.isInteger(bid) || bid <= nomination.currentBid) {
            throw new Error(`A bid must be a whole number of dollars above the current $${nomination.currentBid}.`)
        }
        if (bid > nomination.maxAllowedBids[drafterIndex]) {
            throw new Error(`$${bid} is above ${readTeamIdentity(drafterIndex)}'s maximum allowable bid of `
                            + `$${nomination.maxAllowedBids[drafterIndex]}.`)
        }
        const team = readTeamIdentity(drafterIndex)
        nomination.currentBid = bid
        nomination.highBidderIndex = drafterIndex
        nomination.lastBids[drafterIndex] = bid
        nomination.bidHistory.push({ team, bid, action: 'raise' })
        nomination.turnIndex = (drafterIndex + 1) % getNDrafters()
        continueNomination(nomination)
    } catch (error) {
        reportAutodraftError(error)
    }
}

/** The drafter whose turn it is passes, dropping out of the bidding for this player. */
export function passManualBid(drafterIndex: number): void {
    try {
        const nomination = pendingNomination
        if (!nomination) throw new Error('There is no player up for bid.')
        if (drafterIndex !== nomination.turnIndex) throw new Error(`It is not ${readTeamIdentity(drafterIndex)}'s turn.`)
        if (drafterIndex === nomination.highBidderIndex) {
            throw new Error(`${readTeamIdentity(drafterIndex)} holds the high bid and cannot pass.`)
        }
        nomination.activeBidders.delete(drafterIndex)
        nomination.bidHistory.push({ team: readTeamIdentity(drafterIndex), bid: nomination.currentBid, action: 'pass' })
        nomination.turnIndex = (drafterIndex + 1) % getNDrafters()
        continueNomination(nomination)
    } catch (error) {
        reportAutodraftError(error)
    }
}

/** The manual drafter due to nominate puts a player up for bid at its opening bid. */
export async function nominateManually(
    playerId: number
    , openingBid: number
): Promise<void> {
    try {
        if (pendingNomination || nominationRequestInFlight) throw new Error('A player is already up for bid.')
        if (!Number.isInteger(openingBid) || openingBid < 1) {
            throw new Error(`The opening bid must be a whole number of dollars, at least $1 (got ${openingBid}).`)
        }
        const nominatorIndex = getNominatorIndex()
        if (isAuctionAutodrafter(nominatorIndex)) {
            throw new Error(`${readTeamIdentity(nominatorIndex)} is an autodrafter and nominates on its own.`)
        }
        nominationRequestInFlight = true
        notifyAuctionAutodraftChanged()
        const generation = boardGeneration
        const result = await withSessionRetry(() => nominateAuctionPlayer(requireSessionId(), {
            ...getAuctionState(),
            nominator_id:        readTeamIdentity(nominatorIndex),
            nominated_player_id: playerId,
            opening_bid:         openingBid,
            valuation_team_ids:  listAutodrafterTeamIds(),
        }))
        nominationRequestInFlight = false
        if (generation !== boardGeneration) return
        continueNomination(buildPendingNomination(result, nominatorIndex))
    } catch (error) {
        nominationRequestInFlight = false
        reportAutodraftError(error)
    }
}

/**
 * Applies a drafter's switch between manual and autodraft. Mid-nomination, a drafter switched to autodraft
 * gets its valuation of the player up for bid and the bidding resumes; one switched to manual is prompted
 * on its next turn. Between nominations, the autopilot starts if an autodrafter is now due to nominate.
 */
export function handleAuctionDrafterToggle(drafterIndex: number): void {
    try {
        const nomination = pendingNomination
        if (nomination && isAuctionAutodrafter(drafterIndex)) {
            continueNomination(nomination)
            return
        }
        notifyAuctionAutodraftChanged()
        startAuctionAutopilotIfDue()
    } catch (error) {
        reportAutodraftError(error)
    }
}
