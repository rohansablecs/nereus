from port_operations import calculate_port_operation


tests = [
    {
        "port": "Dhamra",
        "operation": "discharge",
        "cargo_type": "coal",
        "cargo_mt": 60000,
    },
    {
        "port": "Paradip",
        "operation": "discharge",
        "cargo_type": "coking coal",
        "cargo_mt": 60000,
    },
]


for scenario in tests:

    result = calculate_port_operation(
        port=scenario["port"],
        operation=scenario["operation"],
        cargo_type=scenario["cargo_type"],
        cargo_mt=scenario["cargo_mt"],
    )

    print("\n" + "=" * 60)
    print(result.port)
    print("=" * 60)

    print("Operation:", result.operation)
    print("Cargo:", result.cargo_mt, "MT")
    print("Rate:", result.handling_rate_mt_day)
    print("Handling days:", result.handling_days)
    print("Charge:", result.handling_charge_inr)
    print("Waiting:", result.waiting_days)
    print("Status:", result.status)

    if result.reason:
        print("Reason:", result.reason)