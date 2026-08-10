# Futures Contract Forecasting

This context defines the market objects used by the concrete-contract futures forecasting pipeline. Its production and training data exclude roll-stitched continuous contracts.

## Market objects

**Product**:
The commodity root whose delivery-month contracts share an underlying market, such as `rb` or `i`.
_Avoid_: Contract, instrument

**Tradable Contract**:
A concrete delivery-month future, such as `rb2610`, that can be selected for a production forecast and execution. Its usable price history begins at listing and ends when it no longer meets the production liquidity rule.
_Avoid_: Continuous contract, product

**Continuous Contract**:
A provider-produced, roll-stitched series such as `rb8888`. It is outside the concrete-contract production and training corpus.
_Avoid_: Training contract, prediction contract

**Contract Panel**:
The time-aligned prediction contract plus zero or more neighboring concrete contracts that coexist at a forecast time. Each member supplies its own six-dimensional OHLCVA sequence; a panel does not imply a fixed contract count.
_Avoid_: One continuous series

**Neighbor Contract**:
Another active contract of the same product used as contextual input for a prediction contract. Its signed delivery-month distance determines whether it is a near or far neighbor.
_Avoid_: Required third stream, replacement prediction contract

**Closest-Maturity Neighbor**:
The single neighbor used by the first paired experiment: the active same-product contract with the smallest absolute signed delivery-month distance from the prediction contract. If both sides are equally distant, the first version deterministically selects the later-delivery contract.
_Avoid_: Mandatory near-and-far pair, random neighbor

**Near Contract**:
The lower-delivery-maturity member of a contract panel relative to another panel member. It is a maturity relationship, not a continuous-contract roll state.
_Avoid_: Continuous contract, prior contract

**Far Contract**:
The higher-delivery-maturity member of a contract panel relative to another panel member.
_Avoid_: Future continuous contract

**Maturity Role**:
The signed relationship of a neighbor to the prediction contract at a forecast origin: earlier delivery or later delivery. It describes a relationship, not a required model-input slot.
_Avoid_: Price feature, roll event

## Forecasting objects

**Prediction Contract**:
The concrete contract supplied to the forecasting pipeline for one forecast. It is the sole source of that forecast's context, target, and realized evaluation return.
_Avoid_: Product forecast, `8888` forecast

**Contract-local Context**:
The historical OHLCVA bars of one panel member available before the forecast origin. It never contains bars from another contract.
_Avoid_: Continuous history, stitched context

**Panel Forecast Case**:
Time-aligned contract-local contexts from the prediction contract and zero or more neighbor contracts, paired with a future target for the prediction contract. The member streams remain separate rather than being spliced into one price series.
_Avoid_: Stitched sample, continuous-contract case

**Target-only Fallback**:
The separately validated single-contract forecasting path used when no eligible neighbor exists. A pair model is not fed a fabricated or untrained missing-neighbor stream.
_Avoid_: Zero-filled K-line stream, copied target as neighbor

**Complete Panel Snapshot**:
The recorded set of all active concrete contracts for a product at a forecast origin. It is the authority for assigning near and far neighbors in production-equivalent evaluation.
_Avoid_: Current contract list applied retrospectively

**Partial Panel**:
A retrospective panel whose historical active-contract set cannot be fully reconstructed because expired contracts are unavailable. It may support exploratory training but not final panel validation.
_Avoid_: Complete historical panel

**Minimum Context Requirement**:
The predeclared number of observed contract-local bars required for a forecast. A contract with fewer bars has no forecast case; it is not padded with another contract's history.
_Avoid_: Filling history from another contract
