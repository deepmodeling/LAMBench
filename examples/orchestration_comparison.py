#!/usr/bin/env python3
"""
Example usage comparison between different orchestration backends.
"""

print("LAMBench Orchestration Methods:")
print("=" * 50)

print("\n1. Local Execution (no orchestration):")
print("   lambench --local --models DP_2024Q4 --tasks HPt_NC_2022")
print("   - Runs tasks sequentially on local machine")
print("   - No distributed execution or monitoring")
print("   - Best for testing and small workloads")

print("\n2. dflow Orchestration (existing):")
print("   lambench --models DP_2024Q4 --tasks HPt_NC_2022")
print("   - Uses dflow for distributed execution")
print("   - Integrates with Bohrium cloud platform")
print("   - Provides task monitoring and job groups")

print("\n3. Prefect Orchestration (NEW):")
print("   lambench --prefect --models DP_2024Q4 --tasks HPt_NC_2022")  
print("   - Uses Prefect for workflow management")
print("   - Built-in monitoring UI and dashboard")
print("   - Support for Prefect Cloud and on-premises deployment")

print("\n" + "=" * 50)
print("Installation commands:")
print("  pip install lambench[dflow]    # for dflow support")
print("  pip install lambench[prefect]  # for Prefect support")

print("\nFor more details on Prefect orchestration:")
print("  See docs/prefect_orchestration.md")