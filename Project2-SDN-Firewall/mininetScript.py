#!/usr/bin/python3
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

def sdnController():
    #Create 2-Controller, 1-Switch and 4-Host 
    net = Mininet(controller=Controller, switch=OVSSwitch, autoSetMacs=True)

    info('*** Adding controllers\n')
    c1 = net.addController('c1', port=6633)
    c2 = net.addController('c2', port=6655)

    info('*** Adding switch\n')
    s1 = net.addSwitch('s1')


    info('*** Adding hosts\n')
    #Task-1
    #h1 = net.addHost('h1')
    #h2 = net.addHost('h2')
    #h3 = net.addHost('h3')
    #h4 = net.addHost('h4')

    #Task-2
    info('*** Adding hosts\n')
    # Assign IP address 192.168.2.10 to container host #1
    h1 = net.addHost('h1', ip='192.168.2.10/24')
    
    # Assign IP address 192.168.2.20 to container host #2
    h2 = net.addHost('h2', ip='192.168.2.20/24')
    
    # Assign IP address 192.168.2.30 to container host #3
    h3 = net.addHost('h3', ip='192.168.2.30/24')
    
    # Assign IP address 192.168.2.40 to container host #4
    h4 = net.addHost('h4', ip='192.168.2.40/24')


    info('*** Creating links\n')
    net.addLink(s1, h1)
    net.addLink(s1, h2)
    net.addLink(s1, h3)
    net.addLink(s1, h4)

    # Mininet automatically connects controllers to switches
    # but if we want to add explicit links if needed:
    #net.addLink(c1, s1)
    #net.addLink(c2, s1)

    info('*** Starting network\n')
    net.start()
    display_links(net)
    # --- Apply Firewall Rules Here ---
    #l2config_path = "/home/aditya244/Downloads/pox/l2firewall.config"
    l2config_path = "/home/aditya244/Downloads/pox/l2firewall.iptables"
    #l3config_path = "/home/aditya244/Downloads/pox/l3firewall.config"
    l3config_path = "/home/aditya244/Downloads/pox/l3firewall.iptables"


    
    configure_firewall(h1, l2config_path)
    configure_firewall(h2, l2config_path)
    configure_firewall(h3, l2config_path)
    configure_firewall(h4, l2config_path)
    configure_firewall(h1, l3config_path)
    configure_firewall(h2, l3config_path)
    configure_firewall(h3, l3config_path)
    configure_firewall(h4, l3config_path)
    # ----------------------------------
    CLI(net)
    net.stop()

def display_links(net):
    print("\n--- Added links: ---")
    link_strings = []
    for link in net.links:
        node1 = link.intf1.node.name
        node2 = link.intf2.node.name
        link_strings.append(f"({node1}, {node2})")
    
    print(" ".join(link_strings))
    print("----------------------\n")

# Function to apply firewall rules to a given host
def configure_firewall(host, rules_config_file):
    info(f'*** Applying firewall rules to {host.name} from {rules_config_file}\n')
    # Clear existing rules and set default policy to DROP for incoming traffic

    host.cmd('iptables -F')
    #host.cmd('iptables -P INPUT DROP')
    host.cmd('iptables -P INPUT ACCEPT')  
    #host.cmd('iptables -P FORWARD DROP')
    host.cmd('iptables -P FORWARD ACCEPT')
    host.cmd('iptables -P OUTPUT ACCEPT') # Allow all outgoing for simplicity
    host.cmd('iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT')

    # Apply rules from the file using iptables-restore
    try:
        # Assuming the rules file contains iptables-restore format rules
        # You need to ensure the rules file is accessible in the Mininet environment
        host.cmd(f'iptables-restore < {rules_config_file}')
        info(f'*** Rules applied to {host.name}\n')
    except Exception as e:
        info(f'*** Error applying rules to {host.name}: {e}\n')
    
if __name__ == '__main__':
    setLogLevel('info')
    sdnController()
