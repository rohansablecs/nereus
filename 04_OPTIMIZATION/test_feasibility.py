from feasibility import Vessel, evaluate_scenario


vessels = [
    Vessel(
        vessel_id="TEST_PANAMAX_A",
        vessel_class="Panamax",
        dwt_mt=75000,
        loa_m=225,
        beam_m=32,
        draft_m=13.0,
    ),
    Vessel(
        vessel_id="TEST_CAPESIZE_A",
        vessel_class="Capesize",
        dwt_mt=180000,
        loa_m=290,
        beam_m=45,
        draft_m=17.0,
    ),
]


result = evaluate_scenario(
    cargo_mt=60000,
    cargo_type="Coking Coal",
    destination="Paradip",
    vessels=vessels,
)


for vessel in result["vessels"]:
    print("\n" + "=" * 70)
    print(vessel["vessel"]["vessel_id"])
    print("=" * 70)

    print("Class:", vessel["vessel"]["vessel_class"])
    print("Feasible:", vessel["feasible"])
    print("Feasible berths:", vessel["feasible_berths"])

    for berth in vessel["berths"]:
        print(
            f"\n{berth['berth']}: "
            f"{'PASS' if berth['feasible'] else 'FAIL'}"
        )

        if berth["reasons"]:
            print("  Reasons:", ", ".join(berth["reasons"]))

        for check in berth["checks"]:
            print(
                f"  {check['constraint']}: "
                f"{check['status']}"
            )
