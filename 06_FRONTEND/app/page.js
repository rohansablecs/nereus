"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  CircleAlert,
  Compass,
  LoaderCircle,
  Ship,
  Waves,
  X,
} from "lucide-react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const DEFAULT_SCENARIO = {
  cargo: "coking coal",
  quantity_mt: 60000,
  origin_id: "AU_NEWCASTLE",
  destination_id: "IN_PARADIP",
  delivery_window_days: 30,
  contract_strategy: "spot",
};

function extractArray(value, keys = []) {
  if (Array.isArray(value)) return value;

  if (!value || typeof value !== "object") return [];

  for (const key of keys) {
    if (Array.isArray(value[key])) return value[key];
  }

  if (Array.isArray(value.items)) return value.items;
  if (Array.isArray(value.data)) return value.data;
  if (Array.isArray(value.results)) return value.results;

  return [];
}

function findField(item, fields, fallback = "") {
  if (!item || typeof item !== "object") return fallback;

  for (const field of fields) {
    if (
      item[field] !== undefined &&
      item[field] !== null &&
      item[field] !== ""
    ) {
      return item[field];
    }
  }

  return fallback;
}

function formatNumber(value, digits = 0) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatMoney(value, currency) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return `${currency}${Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 0,
  })}`;
}

function titleCase(value) {
  if (!value) return "—";

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getOriginId(item) {
  return findField(item, [
    "origin_id",
    "id",
    "code",
    "originCode",
  ]);
}

function getDestinationId(item) {
  return findField(item, [
    "destination_id",
    "id",
    "code",
    "destinationCode",
  ]);
}

function State({ status }) {
  const normalized = String(status || "").toUpperCase();

  let className = "state state-unknown";
  let icon = <CircleAlert size={13} />;

  if (
    normalized.includes("FEASIBLE") &&
    !normalized.includes("INFEASIBLE") &&
    !normalized.includes("POTENTIALLY")
  ) {
    className = "state state-good";
    icon = <Check size={13} />;
  } else if (
    normalized.includes("AVAILABLE") ||
    normalized.includes("CONFIRMED")
  ) {
    className = "state state-good";
    icon = <Check size={13} />;
  } else if (
    normalized.includes("POTENTIALLY") ||
    normalized.includes("UNKNOWN") ||
    normalized.includes("PARTIAL") ||
    normalized.includes("CONDITIONAL")
  ) {
    className = "state state-watch";
  } else if (normalized.includes("INFEASIBLE")) {
    className = "state state-bad";
    icon = <X size={13} />;
  }

  return (
    <span className={className}>
      {icon}
      {titleCase(status || "unknown")}
    </span>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
  disabled,
}) {
  return (
    <label className="input-field">
      <span>{label}</span>

      <div className="select-wrap">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
        >
          {options.length === 0 && (
            <option value="">Loading…</option>
          )}

          {options.map((option) => {
            const id =
              option.id ||
              option.origin_id ||
              option.destination_id;

            const name =
              option.name ||
              option.origin_name ||
              option.destination_name ||
              option.port_name ||
              id;

            return (
              <option key={id} value={id}>
                {name}
              </option>
            );
          })}
        </select>

        <ChevronDown size={17} />
      </div>
    </label>
  );
}

export default function Home() {
  const [metadata, setMetadata] = useState(null);
  const [metadataError, setMetadataError] = useState("");

  const [scenario, setScenario] =
    useState(DEFAULT_SCENARIO);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  /*
   * =========================================================
   * WHAT-IF STATE
   * =========================================================
   */

  const [whatIfDelivery, setWhatIfDelivery] =
    useState(20);

  const [whatIfBunker, setWhatIfBunker] =
    useState(20);

  const [whatIfResult, setWhatIfResult] =
    useState(null);

  const [whatIfLoading, setWhatIfLoading] =
    useState(false);

  const [whatIfError, setWhatIfError] =
    useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadMetadata() {
      try {
        const response = await fetch(
          `${API_BASE}/api/decision/metadata`
        );

        if (!response.ok) {
          throw new Error(
            `Metadata request failed (${response.status})`
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setMetadata(data);
        }
      } catch (err) {
        if (!cancelled) {
          setMetadataError(err.message);
        }
      }
    }

    loadMetadata();

    return () => {
      cancelled = true;
    };
  }, []);

  const origins = useMemo(
    () =>
      extractArray(metadata, [
        "origins",
        "origin_master",
        "originMaster",
      ]),
    [metadata]
  );

  const destinations = useMemo(
    () =>
      extractArray(metadata, [
        "destinations",
        "destination_master",
        "destinationMaster",
      ]),
    [metadata]
  );

  useEffect(() => {
    if (!origins.length) return;

    if (
      !origins.some(
        (item) =>
          getOriginId(item) === scenario.origin_id
      )
    ) {
      setScenario((current) => ({
        ...current,
        origin_id: getOriginId(origins[0]),
      }));
    }
  }, [origins, scenario.origin_id]);

  useEffect(() => {
    if (!destinations.length) return;

    if (
      !destinations.some(
        (item) =>
          getDestinationId(item) ===
          scenario.destination_id
      )
    ) {
      setScenario((current) => ({
        ...current,
        destination_id:
          getDestinationId(destinations[0]),
      }));
    }
  }, [destinations, scenario.destination_id]);

  /*
   * =========================================================
   * BASE ANALYSIS
   * =========================================================
   */

  async function analyzeScenario(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);
    setWhatIfResult(null);
    setWhatIfError("");

    try {
      const response = await fetch(
        `${API_BASE}/api/decision/analyze`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            cargo: scenario.cargo,
            quantity_mt: Number(
              scenario.quantity_mt
            ),
            origin_id: scenario.origin_id,
            destination_id:
              scenario.destination_id,
            delivery_window_days: Number(
              scenario.delivery_window_days
            ),
            contract_strategy:
              scenario.contract_strategy,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            `Decision request failed (${response.status})`
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err?.message ||
          "Unable to reach the NEREUS decision engine."
      );
    } finally {
      setLoading(false);
    }
  }

  /*
   * =========================================================
   * WHAT-IF ANALYSIS
   * =========================================================
   *
   * Delivery window:
   * Re-runs the REAL deterministic engine.
   *
   * Bunker stress:
   * Applies a transparent sensitivity multiplier to the
   * bunker component returned by the engine.
   *
   * It does NOT pretend the underlying bunker observation
   * has changed.
   */

  async function runWhatIf() {
    setWhatIfLoading(true);
    setWhatIfError("");
    setWhatIfResult(null);

    try {
      const deliveryDays =
        Number(whatIfDelivery);

      const bunkerStress =
        Number(whatIfBunker);

      if (
        !Number.isFinite(deliveryDays) ||
        deliveryDays <= 0
      ) {
        throw new Error(
          "Delivery window must be greater than zero."
        );
      }

      if (
        !Number.isFinite(bunkerStress) ||
        bunkerStress <= -100
      ) {
        throw new Error(
          "Invalid bunker stress value."
        );
      }

      const response = await fetch(
        `${API_BASE}/api/decision/analyze`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            cargo: scenario.cargo,
            quantity_mt: Number(
              scenario.quantity_mt
            ),
            origin_id: scenario.origin_id,
            destination_id:
              scenario.destination_id,
            delivery_window_days:
              deliveryDays,
            contract_strategy:
              scenario.contract_strategy,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            `What-if request failed (${response.status})`
        );
      }

      const whatIfCandidates =
        data?.candidates ||
        data?.decision?.candidates ||
        [];

      const whatIfSummary =
        data?.decision?.best_candidate ||
        null;

      const whatIfBest =
        whatIfCandidates.find(
          (candidate) =>
            candidate.vessel_class ===
            whatIfSummary?.vessel_class
        ) ||
        whatIfSummary ||
        null;

      const baseBunker =
        Number(
          whatIfBest?.economics
            ?.bunker_cost_usd
        ) || 0;

      const stressedBunker =
        baseBunker > 0
          ? baseBunker *
            (1 + bunkerStress / 100)
          : null;

      setWhatIfResult({
        data,
        best: whatIfBest,
        candidates: whatIfCandidates,
        bunker: {
          base: baseBunker,
          stressed: stressedBunker,
          stress_percent: bunkerStress,
        },
        delivery_window_days:
          deliveryDays,
      });
    } catch (err) {
      setWhatIfError(
        err?.message ||
          "Unable to run the what-if scenario."
      );
    } finally {
      setWhatIfLoading(false);
    }
  }

  /*
   * =========================================================
   * DECISION DATA
   * =========================================================
   */

  const decision = result?.decision;

  const candidates =
    result?.candidates ||
    decision?.candidates ||
    [];

  const bestSummary =
    decision?.best_candidate;

  const detailedBest =
    candidates.find(
      (candidate) =>
        candidate.vessel_class ===
        bestSummary?.vessel_class
    ) || bestSummary;

  const best = detailedBest;

  const scenarioData = result?.scenario;

  /*
   * =========================================================
   * AI REASONING
   * =========================================================
   */

  const ai = result?.ai;
  const aiDecision = ai?.decision;

  const aiAvailable =
    ai?.status === "available" &&
    aiDecision;

  /*
   * =========================================================
   * ROUTE TIMING
   * =========================================================
   */

  const confirmedDeliveryDays =
    best?.delivery?.voyage_days ?? null;

  const knownVoyageDays =
    best?.route?.known_voyage_days ?? null;

  const voyageDays =
    confirmedDeliveryDays ??
    knownVoyageDays ??
    null;

  const voyageLabel =
    confirmedDeliveryDays != null
      ? "EST. VOYAGE DAYS"
      : "KNOWN VOYAGE DAYS";

  const candidateLabel =
    best?.feasibility_status ===
    "FEASIBLE"
      ? "SELECTED CANDIDATE"
      : "FEASIBLE CANDIDATE";

  const aiAlternatives =
    Array.isArray(
      aiDecision?.alternatives
    )
      ? aiDecision.alternatives
      : [];

  const aiWhy =
    Array.isArray(aiDecision?.why)
      ? aiDecision.why
      : [];

  const aiRisks =
    Array.isArray(aiDecision?.risks)
      ? aiDecision.risks
      : [];

  const aiMissing =
    Array.isArray(
      aiDecision?.missing_data
    )
      ? aiDecision.missing_data
      : [];

  /*
   * =========================================================
   * WHAT-IF DERIVED DATA
   * =========================================================
   */

  const whatIfData =
    whatIfResult?.data;

  const whatIfAI =
    whatIfData?.ai;

  const whatIfAIDecision =
    whatIfAI?.decision;

  const whatIfDeliveryEstimate =
    whatIfResult?.best?.delivery
      ?.voyage_days ??
    null;

  const whatIfKnownVoyage =
    whatIfResult?.best?.route
      ?.known_voyage_days ??
    null;

  const whatIfVoyage =
    whatIfDeliveryEstimate ??
    whatIfKnownVoyage ??
    null;

  return (
    <main>
      {/* =====================================================
          HERO
          ===================================================== */}

      <section className="hero">
        <div className="hero-content">
          <span className="hero-kicker">
            NEREUS / CHARTERING INTELLIGENCE
          </span>

          <h1>
            Charter with
            <br />
            <em>foresight.</em>
          </h1>

          <p>
            Freight forecasting, vessel feasibility and
            port intelligence brought together into one
            procurement decision.
          </p>

          <div className="hero-meta">
            <span>
              <Waves size={14} />
              EAST COAST INDIA
            </span>

            <span>
              <Compass size={14} />
              LIVE DECISION ENGINE
            </span>
          </div>
        </div>
      </section>

      {/* =====================================================
          SCENARIO
          ===================================================== */}

      <section className="decision-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">
              01 / SCENARIO
            </span>

            <h2>
              Define the
              <br />
              voyage.
            </h2>
          </div>

          <p>
            Give NEREUS the procurement requirement.
            The decision engine evaluates cargo, route,
            vessel constraints and available operational
            evidence.
          </p>
        </div>

        <form
          className="decision-form"
          onSubmit={analyzeScenario}
        >
          <label className="input-field">
            <span>CARGO</span>

            <input
              value={scenario.cargo}
              onChange={(event) =>
                setScenario((current) => ({
                  ...current,
                  cargo: event.target.value,
                }))
              }
            />
          </label>

          <label className="input-field quantity-input">
            <span>QUANTITY</span>

            <div>
              <input
                type="number"
                min="1"
                value={scenario.quantity_mt}
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    quantity_mt:
                      event.target.value,
                  }))
                }
              />

              <b>MT</b>
            </div>
          </label>

          <SelectField
            label="ORIGIN"
            value={scenario.origin_id}
            onChange={(value) =>
              setScenario((current) => ({
                ...current,
                origin_id: value,
              }))
            }
            options={origins}
            disabled={!origins.length}
          />

          <SelectField
            label="DESTINATION"
            value={
              scenario.destination_id
            }
            onChange={(value) =>
              setScenario((current) => ({
                ...current,
                destination_id: value,
              }))
            }
            options={destinations}
            disabled={!destinations.length}
          />

          <label className="input-field quantity-input">
            <span>DELIVERY WINDOW</span>

            <div>
              <input
                type="number"
                min="1"
                value={
                  scenario.delivery_window_days
                }
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    delivery_window_days:
                      event.target.value,
                  }))
                }
              />

              <b>DAYS</b>
            </div>
          </label>

          <label className="input-field">
            <span>
              CONTRACT STRATEGY
            </span>

            <div className="select-wrap">
              <select
                value={
                  scenario.contract_strategy
                }
                onChange={(event) =>
                  setScenario((current) => ({
                    ...current,
                    contract_strategy:
                      event.target.value,
                  }))
                }
              >
                <option value="spot">
                  Spot
                </option>

                <option value="short_term">
                  Short-term
                </option>

                <option value="medium_term">
                  Medium-term
                </option>

                <option value="multiple_voyage">
                  Multiple voyage
                </option>
              </select>

              <ChevronDown size={17} />
            </div>
          </label>

          <button
            className="analyze-button"
            type="submit"
            disabled={
              loading ||
              !scenario.origin_id ||
              !scenario.destination_id ||
              !scenario.quantity_mt
            }
          >
            {loading ? (
              <>
                <LoaderCircle
                  className="spin"
                  size={18}
                />
                ANALYZING
              </>
            ) : (
              <>
                ANALYZE VOYAGE
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        {metadataError && (
          <div className="error-box">
            <CircleAlert size={17} />
            Unable to load metadata:{" "}
            {metadataError}
          </div>
        )}

        {error && (
          <div className="error-box">
            <CircleAlert size={17} />
            {error}
          </div>
        )}
      </section>

      {/* =====================================================
          RESULTS
          ===================================================== */}

      {result && (
        <>
          {/* =================================================
              ROUTE
              ================================================= */}

          <section className="route-result">
            <div className="route-title">
              <span className="eyebrow">
                02 / ROUTE ASSESSMENT
              </span>

              <h2>
                {scenarioData?.origin ||
                  scenario.origin_id}

                <ArrowRight size={34} />

                {scenarioData?.destination ||
                  scenario.destination_id}
              </h2>

              <p>
                {scenarioData?.cargo ||
                  scenario.cargo}{" "}
                ·{" "}
                {formatNumber(
                  scenarioData?.quantity_mt ??
                    scenario.quantity_mt
                )}{" "}
                MT
              </p>
            </div>

            <div className="route-stat">
              <span className="eyebrow">
                DISTANCE
              </span>

              <strong>
                {formatNumber(
                  best?.route?.distance_nm
                )}
              </strong>

              <small>
                NAUTICAL MILES
              </small>
            </div>

            <div className="route-stat">
              <span className="eyebrow">
                {voyageLabel}
              </span>

              <strong>
                {formatNumber(
                  voyageDays,
                  1
                )}
              </strong>

              <small>
                {confirmedDeliveryDays !=
                null
                  ? "DELIVERY ESTIMATE"
                  : "CURRENTLY KNOWN"}
              </small>
            </div>
          </section>

          {/* =================================================
              SNAPSHOT
              ================================================= */}

          <section className="analysis-grid">
            <article className="analysis-card">
              <div className="card-top">
                <div>
                  <span className="eyebrow">
                    {candidateLabel}
                  </span>

                  <h3>
                    Vessel selection
                  </h3>
                </div>

                <Ship />
              </div>

              <div className="big-value">
                {best?.vessel_class ||
                  "—"}
              </div>

              <p>
                {best?.feasibility_status
                  ? titleCase(
                      best.feasibility_status
                    )
                  : "No confirmed candidate"}
              </p>

              <div
                style={{
                  marginTop: 24,
                }}
              >
                <State
                  status={
                    best?.feasibility_status ||
                    "UNKNOWN"
                  }
                />
              </div>
            </article>

            <article className="analysis-card entry-card">
              <div className="card-top">
                <div>
                  <span className="eyebrow">
                    PORT CONDITION
                  </span>

                  <h3>
                    Operational state
                  </h3>
                </div>

                <Waves />
              </div>

              <div className="entry-result">
                <strong>
                  {best?.port
                    ?.pressure_band ||
                    "—"}
                </strong>

                <span>
                  Port pressure{" "}
                  {best?.port?.pressure !=
                  null
                    ? formatNumber(
                        best.port.pressure,
                        2
                      )
                    : "—"}
                </span>
              </div>

              <p>
                Pressure is an operational
                diagnostic signal. It is not
                presented as a calibrated
                waiting-time forecast.
              </p>
            </article>
          </section>

          {/* =================================================
              VESSELS
              ================================================= */}

          <section className="feature-section">
            <div className="feature-heading">
              <div>
                <span className="eyebrow">
                  03 / VESSEL FEASIBILITY
                </span>

                <h2>
                  Every class.
                  <br />
                  One decision.
                </h2>
              </div>

              <p>
                NEREUS evaluates vessel classes
                against cargo capacity and
                destination physical constraints.
              </p>
            </div>

            <div className="vessel-results">
              {candidates.map(
                (candidate) => {
                  const hasDeliveryEstimate =
                    candidate.delivery
                      ?.voyage_days != null;

                  const knownVoyage =
                    candidate.route
                      ?.known_voyage_days !=
                    null;

                  const candidateTiming =
                    hasDeliveryEstimate
                      ? candidate.delivery
                          .voyage_days
                      : knownVoyage
                      ? candidate.route
                          .known_voyage_days
                      : null;

                  return (
                    <article
                      className="vessel-result"
                      key={
                        candidate.vessel_class
                      }
                    >
                      <div className="vessel-icon">
                        <Ship size={24} />
                      </div>

                      <div className="vessel-main">
                        <span>
                          VESSEL CLASS
                        </span>

                        <h3>
                          {
                            candidate.vessel_class
                          }
                        </h3>

                        <p>
                          {candidate.reason ||
                            candidate.feasibility_status ||
                            "No additional explanation available."}
                        </p>
                      </div>

                      <div className="vessel-constraint">
                        <span>
                          {hasDeliveryEstimate
                            ? "DELIVERY"
                            : knownVoyage
                            ? "KNOWN VOYAGE"
                            : "DELIVERY"}
                        </span>

                        <strong>
                          {candidateTiming !=
                          null
                            ? `${formatNumber(
                                candidateTiming,
                                1
                              )} d`
                            : "—"}
                        </strong>
                      </div>

                      <div className="vessel-result-state">
                        <State
                          status={
                            candidate.feasibility_status
                          }
                        />
                      </div>
                    </article>
                  );
                }
              )}
            </div>
          </section>

          {/* =================================================
              PORT
              ================================================= */}

          <section className="dark-section">
            <div className="feature-heading">
              <div>
                <span className="eyebrow">
                  04 / PORT INTELLIGENCE
                </span>

                <h2>
                  The berth is part
                  <br />
                  of the decision.
                </h2>
              </div>

              <p>
                Physical compatibility and current
                operational evidence are evaluated
                together.
              </p>
            </div>

            <div className="port-analysis">
              <div className="port-side">
                <span>
                  DESTINATION
                </span>

                <h3>
                  {scenarioData?.destination ||
                    scenario.destination_id}
                </h3>

                <div className="port-constraints">
                  <div>
                    <span>
                      PRESSURE
                    </span>

                    <strong>
                      {best?.port?.pressure !=
                      null
                        ? formatNumber(
                            best.port.pressure,
                            2
                          )
                        : "—"}
                    </strong>
                  </div>

                  <div>
                    <span>BAND</span>

                    <strong>
                      {best?.port
                        ?.pressure_band ||
                        "—"}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="port-connector">
                <ArrowRight size={24} />
              </div>

              <div className="port-side">
                <span>
                  BERTH ASSESSMENT
                </span>

                <h3>
                  {best?.berths
                    ?.feasible_count ??
                    best?.feasible_berths
                      ?.length ??
                    "—"}

                  <small className="port-small-label">
                    CONFIRMED FEASIBLE BERTHS
                  </small>
                </h3>

                <div className="port-constraints">
                  <div>
                    <span>
                      EVALUATED
                    </span>

                    <strong>
                      {best?.berths
                        ?.evaluated ??
                        "—"}
                    </strong>
                  </div>

                  <div>
                    <span>
                      UNKNOWN
                    </span>

                    <strong>
                      {best?.berths
                        ?.unknown_count ??
                        "—"}
                    </strong>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* =================================================
              ECONOMICS
              ================================================= */}

          <section className="economics-section">
            <div className="feature-heading">
              <div>
                <span className="eyebrow">
                  05 / VOYAGE ECONOMICS
                </span>

                <h2>
                  What the voyage
                  <br />
                  costs today.
                </h2>
              </div>

              <p>
                Only cost components supported by
                the current data foundation are
                displayed.
              </p>
            </div>

            <div className="cost-grid">
              <div className="cost">
                <span>
                  BUNKER COST
                </span>

                <strong>
                  {formatMoney(
                    best?.economics
                      ?.bunker_cost_usd,
                    "$"
                  )}
                </strong>
              </div>

              <div className="cost">
                <span>
                  HANDLING
                </span>

                <strong>
                  {formatMoney(
                    best?.economics
                      ?.handling_charge_inr,
                    "₹"
                  )}
                </strong>
              </div>

              <div className="cost">
                <span>
                  COST STATUS
                </span>

                <strong>
                  {titleCase(
                    best?.economics
                      ?.cost_status ||
                      "partial"
                  )}
                </strong>
              </div>

              <div className="cost">
                <span>
                  SEA TIME
                </span>

                <strong>
                  {best?.route?.sea_days !=
                  null
                    ? `${formatNumber(
                        best.route.sea_days,
                        1
                      )} d`
                    : "—"}
                </strong>
              </div>

              <div className="cost">
                <span>
                  KNOWN VOYAGE
                </span>

                <strong>
                  {best?.route
                    ?.known_voyage_days !=
                  null
                    ? `${formatNumber(
                        best.route
                          .known_voyage_days,
                        1
                      )} d`
                    : "—"}
                </strong>
              </div>

              <div className="cost total">
                <span>
                  DELIVERY WINDOW
                </span>

                <strong>
                  {scenarioData
                    ?.delivery_window_days ??
                    scenario.delivery_window_days}{" "}
                  d
                </strong>
              </div>
            </div>
          </section>

          {/* =================================================
              OPERATIONAL EVIDENCE
              ================================================= */}

          <section className="feature-section">
            <div className="feature-heading">
              <div>
                <span className="eyebrow">
                  06 / OPERATIONAL EVIDENCE
                </span>

                <h2>
                  What NEREUS
                  <br />
                  actually knows.
                </h2>
              </div>

              <p>
                Missing signals remain missing.
                They are never converted into zero
                or artificial certainty.
              </p>
            </div>

            <div className="risk-list">
              <div className="risk-row">
                <strong>
                  Operational source
                </strong>

                <span>
                  {best?.operations
                    ?.source_type || "—"}
                </span>

                <State
                  status={
                    best?.operations
                      ?.source_type
                      ? "AVAILABLE"
                      : "UNKNOWN"
                  }
                />
              </div>

              <div className="risk-row">
                <strong>
                  Dry bulk vessels waiting
                </strong>

                <span>
                  {best?.operations
                    ?.state
                    ?.dry_bulk_vessels_waiting !=
                  null
                    ? formatNumber(
                        best.operations.state
                          .dry_bulk_vessels_waiting
                      )
                    : "—"}
                </span>

                <State
                  status={
                    best?.operations
                      ?.state
                      ?.dry_bulk_vessels_waiting !=
                    null
                      ? "AVAILABLE"
                      : "UNKNOWN"
                  }
                />
              </div>

              <div className="risk-row">
                <strong>
                  Berth cycle prediction
                </strong>

                <span>
                  {best?.operations
                    ?.berth_cycle
                    ?.predicted_berth_cycle_days !=
                  null
                    ? `${formatNumber(
                        best.operations
                          .berth_cycle
                          .predicted_berth_cycle_days,
                        1
                      )} days`
                    : "—"}
                </span>

                <State
                  status={
                    best?.operations
                      ?.berth_cycle
                      ?.status ||
                    "UNKNOWN"
                  }
                />
              </div>

              <div className="risk-row">
                <strong>
                  Data age
                </strong>

                <span>
                  {best?.operations
                    ?.data_age_hours !=
                  null
                    ? `${formatNumber(
                        best.operations
                          .data_age_hours,
                        1
                      )} hours`
                    : "—"}
                </span>

                <State
                  status={
                    best?.operations
                      ?.data_age_hours !=
                    null
                      ? "AVAILABLE"
                      : "UNKNOWN"
                  }
                />
              </div>
            </div>
          </section>

          {/* =================================================
              AI DECISION
              ================================================= */}

          <section className="risk-section">
            <div>
              <span className="eyebrow">
                07 / NEREUS DECISION
              </span>

              <h2>
                Why this
                <br />
                charter?
              </h2>

              <p>
                The deterministic engine establishes
                physical feasibility and available
                evidence. NEREUS AI interprets those
                results and explains the resulting
                procurement decision.
              </p>

              {!aiAvailable && (
                <div className="ai-unavailable">
                  AI REASONING LAYER{" "}
                  {ai?.status === "error"
                    ? "ERROR"
                    : "UNAVAILABLE"}
                  <br />
                  The deterministic NEREUS decision
                  remains available.
                </div>
              )}
            </div>

            {aiAvailable ? (
              <div className="ai-stack">
                <div className="risk-row ai-primary">
                  <div className="ai-primary-top">
                    <div>
                      <span className="eyebrow">
                        RECOMMENDED VESSEL
                      </span>

                      <strong className="ai-vessel">
                        {aiDecision.decision ||
                          "No decision"}
                      </strong>
                    </div>

                    <State
                      status={
                        aiDecision
                          .decision_status ||
                        "NO_DECISION"
                      }
                    />
                  </div>

                  <p>
                    {aiDecision.summary ||
                      "No AI explanation was returned."}
                  </p>
                </div>

                {aiWhy.length > 0 && (
                  <div>
                    <span className="eyebrow">
                      WHY
                    </span>

                    <div className="risk-list ai-list">
                      {aiWhy.map(
                        (reason, index) => (
                          <div
                            className="risk-row"
                            key={`why-${index}`}
                          >
                            <strong>
                              {String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )}
                            </strong>

                            <span>
                              {reason}
                            </span>

                            <Check size={15} />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {aiRisks.length > 0 && (
                  <div>
                    <span className="eyebrow">
                      RISKS
                    </span>

                    <div className="risk-list ai-list">
                      {aiRisks.map(
                        (risk, index) => (
                          <div
                            className="risk-row"
                            key={`risk-${index}`}
                          >
                            <strong>
                              {String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )}
                            </strong>

                            <span>
                              {risk}
                            </span>

                            <CircleAlert
                              size={15}
                            />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {aiAlternatives.length >
                  0 && (
                  <div>
                    <span className="eyebrow">
                      ALTERNATIVES
                    </span>

                    <div className="risk-list ai-list">
                      {aiAlternatives.map(
                        (
                          alternative,
                          index
                        ) => (
                          <div
                            className="risk-row"
                            key={`alternative-${index}`}
                          >
                            <strong>
                              {alternative.vessel_class ||
                                "—"}
                            </strong>

                            <span>
                              {alternative.reason ||
                                "No explanation available."}
                            </span>

                            <State
                              status={
                                alternative.status ||
                                "UNKNOWN"
                              }
                            />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {aiMissing.length > 0 && (
                  <div>
                    <span className="eyebrow">
                      MISSING DATA
                    </span>

                    <div className="risk-list ai-list">
                      {aiMissing.map(
                        (
                          missing,
                          index
                        ) => (
                          <div
                            className="risk-row"
                            key={`missing-${index}`}
                          >
                            <strong>
                              {String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )}
                            </strong>

                            <span>
                              {missing}
                            </span>

                            <CircleAlert
                              size={15}
                            />
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                <div className="model-footer">
                  REASONING MODEL{" "}
                  {ai.model ||
                    "HUGGING FACE"}
                  <span>/</span>
                  DETERMINISTIC ENGINE REMAINS
                  AUTHORITATIVE
                </div>
              </div>
            ) : (
              <div className="risk-list">
                <div className="risk-row">
                  <strong>AI</strong>

                  <span>
                    AI reasoning is currently
                    unavailable. The deterministic
                    decision engine remains
                    authoritative and operational.
                  </span>

                  <State status="UNAVAILABLE" />
                </div>
              </div>
            )}
          </section>

          {/* =================================================
              WHAT-IF SIMULATOR
              ================================================= */}

          <section className="scenario-section">
            <div className="feature-heading">
              <div>
                <span className="eyebrow">
                  08 / WHAT-IF SIMULATOR
                </span>

                <h2>
                  Stress the
                  <br />
                  decision.
                </h2>
              </div>

              <p>
                Change the procurement conditions and
                let NEREUS re-run the decision engine
                against the altered delivery requirement.
              </p>
            </div>

            <div className="scenario-layout">
              <div className="scenario-panel">
                <div className="scenario-control">
                  <span>
                    DELIVERY WINDOW
                  </span>

                  <p>
                    Re-evaluates vessel feasibility
                    against a tighter or wider
                    delivery requirement.
                  </p>

                  <div className="scenario-number">
                    <input
                      type="number"
                      min="1"
                      value={whatIfDelivery}
                      onChange={(event) =>
                        setWhatIfDelivery(
                          event.target.value
                        )
                      }
                    />

                    <b>DAYS</b>
                  </div>
                </div>

                <div className="scenario-control">
                  <span>
                    BUNKER PRICE STRESS
                  </span>

                  <p>
                    Economic sensitivity applied
                    to the currently supported
                    bunker component.
                  </p>

                  <div className="scenario-range">
                    <input
                      type="range"
                      min="-30"
                      max="50"
                      step="5"
                      value={whatIfBunker}
                      onChange={(event) =>
                        setWhatIfBunker(
                          event.target.value
                        )
                      }
                    />

                    <strong>
                      {Number(
                        whatIfBunker
                      ) > 0
                        ? "+"
                        : ""}
                      {whatIfBunker}%
                    </strong>
                  </div>
                </div>

                <button
                  className="scenario-run"
                  type="button"
                  onClick={runWhatIf}
                  disabled={
                    whatIfLoading
                  }
                >
                  {whatIfLoading ? (
                    <>
                      <LoaderCircle
                        className="spin"
                        size={17}
                      />
                      RUNNING SCENARIO
                    </>
                  ) : (
                    <>
                      RUN WHAT-IF
                      <ArrowRight size={17} />
                    </>
                  )}
                </button>

                <div className="scenario-note">
                  Delivery-window scenarios are
                  re-evaluated by the deterministic
                  engine. Bunker stress is an economic
                  sensitivity test only.
                </div>
              </div>

              <div className="scenario-output">
                {!whatIfResult &&
                  !whatIfError && (
                    <div className="scenario-empty">
                      <span>
                        SCENARIO READY
                      </span>

                      <p>
                        Alter the delivery window
                        or bunker assumption, then
                        run the scenario.
                      </p>
                    </div>
                  )}

                {whatIfError && (
                  <div className="scenario-empty scenario-error">
                    <span>
                      SCENARIO ERROR
                    </span>

                    <p>
                      {whatIfError}
                    </p>
                  </div>
                )}

                {whatIfResult && (
                  <>
                    <div className="scenario-result-head">
                      <div>
                        <span>
                          SCENARIO RESULT
                        </span>

                        <h3>
                          {whatIfAIDecision
                            ?.decision ||
                            whatIfResult
                              ?.best
                              ?.vessel_class ||
                            "NO DECISION"}
                        </h3>
                      </div>

                      <State
                        status={
                          whatIfAIDecision
                            ?.decision_status ||
                          whatIfResult?.best
                            ?.feasibility_status ||
                          "UNKNOWN"
                        }
                      />
                    </div>

                    <div className="scenario-metrics">
                      <div>
                        <span>
                          DELIVERY WINDOW
                        </span>

                        <strong>
                          {
                            whatIfResult.delivery_window_days
                          }
                          <small> d</small>
                        </strong>
                      </div>

                      <div>
                        <span>
                          DELIVERY ESTIMATE
                        </span>

                        <strong>
                          {whatIfVoyage !=
                          null
                            ? formatNumber(
                                whatIfVoyage,
                                1
                              )
                            : "—"}
                          {whatIfVoyage !=
                            null && (
                            <small> d</small>
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          BUNKER STRESS
                        </span>

                        <strong>
                          {Number(
                            whatIfBunker
                          ) > 0
                            ? "+"
                            : ""}
                          {whatIfBunker}
                          <small>%</small>
                        </strong>
                      </div>

                      <div>
                        <span>
                          STRESSED BUNKER
                        </span>

                        <strong>
                          {whatIfResult
                            .bunker
                            .stressed !=
                          null
                            ? formatMoney(
                                whatIfResult
                                  .bunker
                                  .stressed,
                                "$"
                              )
                            : "—"}
                        </strong>
                      </div>
                    </div>

                    <div className="scenario-summary">
                      <span>
                        ENGINE INTERPRETATION
                      </span>

                      <p>
                        {whatIfAIDecision
                          ?.summary ||
                          "The deterministic engine returned no AI summary."}
                      </p>
                    </div>

                    <div className="scenario-comparison">
                      <div>
                        <span>
                          BASE CASE
                        </span>

                        <strong>
                          {best?.vessel_class ||
                            "—"}
                        </strong>

                        <small>
                          {best
                            ?.feasibility_status ||
                            "UNKNOWN"}
                        </small>
                      </div>

                      <div className="scenario-arrow">
                        →
                      </div>

                      <div>
                        <span>
                          WHAT-IF
                        </span>

                        <strong>
                          {whatIfResult
                            ?.best
                            ?.vessel_class ||
                            "—"}
                        </strong>

                        <small>
                          {whatIfResult
                            ?.best
                            ?.feasibility_status ||
                            "UNKNOWN"}
                        </small>
                      </div>
                    </div>

                    {whatIfAIDecision
                      ?.risks?.length >
                      0 && (
                      <div className="scenario-risks">
                        <span>
                          SCENARIO RISKS
                        </span>

                        <ul>
                          {whatIfAIDecision.risks
                            .slice(0, 3)
                            .map(
                              (
                                risk,
                                index
                              ) => (
                                <li
                                  key={
                                    index
                                  }
                                >
                                  {risk}
                                </li>
                              )
                            )}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </section>

          {/* =================================================
              LIMITATIONS
              ================================================= */}

          {result.limitations?.length >
            0 && (
            <section className="risk-section">
              <div>
                <span className="eyebrow">
                  DECISION CONTEXT
                </span>

                <h2>
                  Read the
                  <br />
                  caveats.
                </h2>

                <p>
                  These limitations are returned
                  directly by the decision engine.
                </p>
              </div>

              <div className="risk-list">
                {result.limitations.map(
                  (
                    limitation,
                    index
                  ) => (
                    <div
                      className="risk-row"
                      key={`${index}-${limitation}`}
                    >
                      <strong>
                        {String(
                          index + 1
                        ).padStart(
                          2,
                          "0"
                        )}
                      </strong>

                      <span>
                        {limitation}
                      </span>

                      <CircleAlert
                        size={15}
                      />
                    </div>
                  )
                )}
              </div>
            </section>
          )}
        </>
      )}

      {!result && !loading && (
        <section className="empty-home">
          <Ship size={42} />

          <h2>
            Awaiting voyage parameters.
          </h2>

          <p>
            Define a cargo movement above and
            NEREUS will evaluate the route,
            vessel classes, port constraints
            and available operational evidence.
          </p>
        </section>
      )}

      {loading && (
        <section className="empty-home">
          <LoaderCircle
            className="spin"
            size={42}
          />

          <h2>
            Analyzing the voyage.
          </h2>

          <p>
            Evaluating route feasibility,
            vessel constraints and current
            operational evidence.
          </p>
        </section>
      )}

      <footer className="site-footer">
        <span>
          NEREUS / FREIGHT INTELLIGENCE
        </span>

        <span>{API_BASE}</span>
      </footer>
    </main>
  );
}