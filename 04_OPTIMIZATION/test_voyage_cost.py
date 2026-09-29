from voyage_cost import (
    VoyageInput,
    calculate_voyage_cost,
)


voyage = VoyageInput(
    route_id="AUS_NEW_PAR",
    vessel_class="Panamax",
    cargo_mt=60000,
    fuel_price_usd_mt=645.50,
    destination_port="Paradip",
    cargo_type="Coking Coal",
)


result = calculate_voyage_cost(voyage)


print("=== NEREUS VOYAGE ECONOMICS ===")
print()

print("Route:", result.route_id)
print("Vessel:", result.vessel_class)
print("Destination:", result.destination_port)
print("Status:", result.status)

if result.reason:
    print("Reason:", result.reason)

print()

print("SEA PASSAGE")
print("Distance:", result.distance_nm, "NM")
print("Speed:", result.speed_kn, "kn")
print("Laden days:", result.laden_days)
print("Laden fuel:", result.laden_fuel_mt, "MT")
print("Bunker cost:", result.bunker_cost_usd, "USD")

print()

print("PORT OPERATION")
print(
    "Discharge rate:",
    result.discharge_handling_rate_mt_day,
    "MT/day",
)
print(
    "Discharge handling days:",
    result.discharge_handling_days,
)
print(
    "Discharge charge:",
    result.discharge_charge_inr,
    "INR",
)
print(
    "Waiting days:",
    result.waiting_days,
)

print()

print("TOTAL")
print(
    "Known voyage days:",
    result.known_voyage_days,
)
print(
    "Known cost:",
    result.known_cost_usd,
    "USD",
)

print()

print("BREAKDOWN")
print(result.breakdown)