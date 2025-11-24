
import subprocess
from collections import defaultdict
import os
import sys

# Path to your l2firewall.config file
L2_CONFIG_PATH = "/home/ubuntu/pox/l2firewall.config"

def monitor_flows():
    """
    Monitors flows on switch s1 and detects MACs using multiple IPs.
    Updates l2firewall.config and reapplies new flow rules if needed.
    """

    print("[INFO] Running ovs-ofctl dump-flows s1 ...")

    try:
        output = subprocess.check_output(["ovs-ofctl", "dump-flows", "s1"]).decode()
    except subprocess.CalledProcessError as e:
        print("[ERROR] Failed to run ovs-ofctl:", e)
        return
    except FileNotFoundError:
        print("[ERROR] ovs-ofctl command not found. Is Open vSwitch installed?")
        return

    mac_to_ips = defaultdict(set)

    # Parse output for MAC and IP addresses
    for line in output.splitlines():
        if "dl_src=" in line and "nw_src=" in line:
            parts = line.split(",")
            mac = None
            ip = None
            for part in parts:
                if part.startswith("dl_src="):
                    mac = part.split("=")[1]
                elif part.startswith("nw_src="):
                    ip = part.split("=")[1]
            if mac and ip:
                mac_to_ips[mac].add(ip)

    # Detect MACs with multiple IPs
    updated = False
    for mac, ips in mac_to_ips.items():
        if len(ips) > 1:
            print(f"[ALERT] MAC {mac} has multiple IPs: {ips}")

            # Append to l2firewall.config
            if os.path.exists(L2_CONFIG_PATH):
                rule_id = len(open(L2_CONFIG_PATH).readlines())
            else:
                rule_id = 1

            with open(L2_CONFIG_PATH, "a") as f:
                for ip in ips:
                    f.write("{},{},any\n".format(rule_id, mac))
                    rule_id += 1

            updated = True

    if updated:
        print(f"[INFO] Updated {L2_CONFIG_PATH} with new blocked MACs.")

        # Optional: Reapply new flow rules if your controller supports it
        try:
            subprocess.call(["ovs-ofctl", "del-flows", "s1"])
            print("[INFO] Cleared old flows.")
            # The controller will push new rules automatically, or you can restart POX.
        except Exception as e:
            print("[WARNING] Could not reapply flow rules:", e)
    else:
        print("[INFO] No MAC with multiple IPs found.")
        

if __name__ == "__main__":
    monitor_flows()
