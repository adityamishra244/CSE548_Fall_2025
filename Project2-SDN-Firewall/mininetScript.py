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
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    h3 = net.addHost('h3')
    h4 = net.addHost('h4')

    info('*** Creating links\n')
    net.addLink(s1, h1)
    net.addLink(s1, h2)
    net.addLink(s1, h3)
    net.addLink(s1, h4)

    # Mininet automatically connects controllers to switches
    # but if we want to add explicit links if needed:
    net.addLink(c1, s1)
    net.addLink(c2, s1)

    info('*** Starting network\n')
    net.start()
    #net.build()
    #c1.start()
    #c2.start()
    #s1.start([c1])

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    sdnController()
