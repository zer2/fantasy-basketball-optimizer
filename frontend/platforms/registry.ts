// platforms/registry.ts
// Frontend connector registry — the counterpart to the backend's integration registry,
// keyed by the same platform labels.

import { PlatformConnector, ConnectorFactory } from './connector.js'
import { makeFantraxConnector } from './fantrax_connector.js'
import { makeYahooConnector } from './yahoo_connector.js'
import { makeEspnConnector } from './espn_connector.js'

const CONNECTOR_FACTORIES: Record<string, ConnectorFactory> = {
    'Retrieve from Fantrax': makeFantraxConnector,
    'Retrieve from Yahoo':   makeYahooConnector,
    'Retrieve from ESPN':    makeEspnConnector,
}

/** The live platform labels the frontend has connectors for. */
export function connectorPlatforms(): string[] {
    return Object.keys(CONNECTOR_FACTORIES)
}

/** Instantiate every connector (sharing the given status callback). */
export function makeConnectors(setStatus: (message: string) => void): PlatformConnector[] {
    return Object.values(CONNECTOR_FACTORIES).map(make => make(setStatus))
}
