# -*- coding: utf-8 -*-
"""
=== Malhotra, Pramodh Kumar, and Maheshwari (MPM) algorithm ===
The algorithm has time complexity of (n^3). The algorithm operates in stages:
At each stage we:
    
    1. construct the network N(f) [residual network] and from it we find the
    auxiliary[layered] network AN(f).
    2. we find a maximal flow g in AN(f)
    3. update flows in the main network

- Construction of AN(f):
    
    Please see book: Combinatorial Optimization, Papadimitriou & Steiglitz,
                        pages 203-206.
    
    The construction of the layered network is the same as in the Dinic algo-
    rth. So you can also check how Dinic's algorithm works (there is a wikipe-
    dia article) 

- Finding maximal flow g in AN(f):
    
    This is the main part of the algorithm.
    
    Check the original paper: alhotra, V. M, M. P Kumar, S.N Maheshwari, 
    “An O(|V|3) algorithm for finding maximum flows in networks”, 
    Inf. Proc. Letters, 7, no. 6 (October 1978), 277 – 78
    
    Detailed  description also on page 206-208 in the Papadimitrou & Steiglitz 
    book.
    
- Updating flows in the main network:
    In Papadimitriou & Steiglitz book page 206 last 4 sentences in the page
    
=== USAGE ===

the mpm(sink, source, network) is the method to compute the maximum flow.
I represent the network as a dictionary of dictionaries i.e.:
{ 
0: {1: {'cap': 5, 'flow': 0}, 2: {'cap': 100, 'flow': 10}},
1: {4: {'cap: 3, 'flow': 2}}
}
The above represents a network consisting of nodes 0, 1, 2,  4
and the following arcs startNode--->endNone (flow/capacity):
0---->1 (0/5)
0---->2 (10/100)
1---->4 (2/3)

I decided to store graphs in files of the format:
start    end    capacity [flow]

flow is optional. So for the above graph the corresponding file is:
0 1 5
0 2 100 10
1 4 3 2

To run some examples do:
python maximum_flow network_file source sink [LOGLEVEL] [LOGFILE]

LOGLEVEL is OPTIONAL and can be: WARNING, INFO, ERROR, DEBUG (DEFAULT=INFO)
IF LOGFILE is specified the output is logged to the specified file else to 
the stdout.

=== LICENCE ====
Copyright (c) <2013>, <Giorgos Komninos>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import sys
import logging
from collections import deque
import pprint
import cStringIO as StringIO


FORMAT = '%(asctime)-15s %(levelname)s - %(message)s'


def _to_str(network):
    out = StringIO.StringIO()
    pp = pprint.PrettyPrinter(indent=1, stream=out)
    pp.pprint(network)
    to_return = out.getvalue()
    out.close()
    return to_return
    

def read_network(f=sys.stdin):
    N = {}
    lines = f.readlines()
    for line in lines:
        s = line.split(' ')
        if len(s) == 4:
            u, v, c, f = [int(e) for e in s]
        else:
            u, v, c = [int(e) for e in s]
            f = 0
        if u not in N:
            N[u] = {}
        if v not in N:
            N[v] = {}
        N[u][v] = {'cap': c, 'flow': f}
    return N


def delete_node(node, network):
    for u, v in network.items():
        if node in v:
            logging.debug('Deleting edge: (%d, %d)', u, node)
            del v[node]
        if node in network:
            logging.debug('Removing node %d from network', node)
            del network[node]


def build_residual_graph(source, sink, network):
    logging.debug('Building residual graph')
    nr = {}
    que = deque()
    que.append(source)
    visited = set()
    visited.add(source)
    while len(que) > 0:
        now = que.popleft()
        logging.debug('Processing neigbors of node %d', now)
        for e in network[now]:
            logging.debug('edge(%d, %d)', now, e)
            r = network[now][e]['cap'] - network[now][e]['flow']
            logging.debug('residual cap is %d', r)
            if now not in nr:
                nr[now] = {}
            if e not in nr:
                nr[e] = {}
            if r > 0:
                nr[now][e] = {'cap': r ,'direction': 'F'}
                logging.debug('adding (%d, %d) with cap = %d to na', now, e, r) 
            if network[now][e]['flow'] > 0:
                nr[e][now] = {'cap': network[now][e]['flow'], 'direction': 'B'}
                logging.debug('adding (%d, %d) with cap = %d to na', e, now,
                              network[now][e]['flow'])
            if e not in visited:
                que.append(e)
            visited.add(e)
    logging.info('Residual network:\n%s', _to_str(nr))
    return nr
 
 
def build_auxiliary(source, sink, network):
    logging.info('Building auxiliary')
    na = {}
    que = deque()
    que.append(source)
    vl = {source: 0} # vertex level
    visited = set()
    visited.add(source)
    while len(que) > 0:
        now = que.popleft()
        logging.debug('Processing neigbors of node %d %s', now, 
                      network[now].keys())
        na[now] = {}
        for e in network[now]:
            if e in vl and e != sink:
                continue
            logging.debug('edge(%d, %d)', now, e)
            logging.debug('adding (%d, %d) to aux', now, e)
            na[now][e] = {'cap': network[now][e]['cap'], 
                          'direction': network[now][e]['direction']}
            vl[e] = vl[now] + 1
            if e not in visited:
                que.append(e)
            visited.add(e)
            
    logging.debug('before: %s', repr(na))
    logging.debug('node layers: %s', repr(vl))
    if sink not in na:
        logging.debug('Sink not in na')
        return None
    sink_level = vl[sink]
    logging.debug('removing nodes with level >= %d (except sink node = %d)', 
                  sink_level, sink)
    complete = False
    for node in [k for k in vl if vl[k] >= sink_level]:
        if node == sink:
            complete = True
            continue
        logging.debug('We should delete node: %d', node)
        delete_node(node, na)
    logging.info('Auxiliary network:\n%s', _to_str(na))
    return na if complete else None
    
    
def build_level_graph(source, sink, network):
    nr = build_residual_graph(source, sink, network)
    na = build_auxiliary(source, sink, nr)
    return na


def calc_throughput(source, sink, auxiliary):
    throughput = {}
    for n, neibors in auxiliary.iteritems():
        if n == source:
            in_cap = sys.maxint
        else:
            in_cap = sum([v[n]['cap'] for u, v in auxiliary.iteritems() 
                      if n in v])
        if n == sink:
            out_cap = sys.maxint
        else:
            out_cap = sum([v['cap'] for _, v in neibors.iteritems()])
        
        throughput[n] = [in_cap, out_cap]
        logging.debug('Throughput[%d]=min(%d, %d)=%d', n, in_cap, out_cap,
                     min(in_cap, out_cap))
        
    return throughput
 
 
def delete_zero_throughput(source, sink, auxiliary, throughput):
    while True:
        has_zero = False
        for node, cap in throughput.items():
            in_cap, out_cap = cap
            thr = min(in_cap, out_cap)
            if thr == 0:
                if node == source or node == sink:
                    logging.info('Node %d (sink | source) has 0 throughput',
                                  node)
                    return False
                has_zero = True
                logging.debug('Node %d has 0 throughput. Should be deleted',
                              node)
                out_to_update = [(u, d['cap']) for u, d in auxiliary[node].items()]
                for n, v in out_to_update:
                    logging.debug('Updating incap (%d) of node %d', 
                                  throughput[n][0], n)
                    throughput[n][0] -= v
                    logging.debug('New incap is %d', throughput[n][0])
                    
                in_to_update = [(u, d[node]['cap']) for u, d in auxiliary.iteritems() 
                                if node in d]
                for n, v in in_to_update:
                    logging.debug('Updating outcap (%d) of node %d',
                                  throughput[n][1], n)
                    throughput[n][1] -= v
                delete_node(node, auxiliary)
                del throughput[node]
        if not has_zero:
            break
    return True


def push(y, h, auxiliary, throughput, g):
    logging.info('Pushing %d unit from %d', h, y)
    q = deque()
    q.append(y)
    req = {u: 0 for u in auxiliary.keys() if u != y}
    req[y] = h
    flows = []
    while len(q) > 0:
        v = q.popleft()
        logging.debug('Doin %d', v)
        for n in auxiliary[v].keys():
            logging.debug(n)
            logging.debug('%s: %s', v, _to_str(auxiliary[v].keys()))
            if req[v] == 0:
                break
            if 'used' in auxiliary[v][n]:
                logging.info('(%d, %d) is used')
                continue
            m = min(auxiliary[v][n]['cap'], req[v])
            auxiliary[v][n]['cap'] -= m
            logging.debug('New capacity of (%d, %d) is %d', 
                          v, n, auxiliary[v][n]['cap'])
            if auxiliary[v][n]['cap'] == 0:
                logging.debug('Removing (%d, %d) from auxiliary', v, n)
                auxiliary[v][n]['used'] = True
                out_to_update = [u for u, d in auxiliary[v].items()]
                for nn in out_to_update:
                    throughput[nn][0] -= m
            req[v] -= m
            req[n] += m
            logging.debug('Appending %d to queue', n)
            q.append(n)
            direction = auxiliary[v][n]['direction']
            if direction == 'B':
                start, end = n, v
                #v, n = n, v
                m = (-1) * m
            else:
                start, end = v, n
            if start not in g:
                g[start] = {}
            if end not in g[start]:
                g[start][end] = 0
            g[start][end] += m
            flows.append('(%d, %d) = %d %s' %(start, end, g[start][end], direction))
            logging.debug('Flow (%d, %d) is %d changed by %d direction %s'
                          , start, end, g[start][end], m, direction)
    logging.info('Push is done. Flows added:\n%s', _to_str(flows))


def pull(s, y, h, auxiliary, throughput, g):
    logging.info('Pulling %d unit to %d', h, y)
    q = deque([y])
    req = {u: 0 for u in auxiliary.keys() if u != y}
    req[y] = h
    flows = []
    while q:
        v = q.popleft()
        for u, d in auxiliary.iteritems():
            if req[v] == 0:
                break
            if v in d:
                if 'used' in auxiliary[u][v]:
                    logging.info('(%d, %d) is used', u, v)
                    continue 
                m = min(auxiliary[u][v]['cap'], req[v])
                logging.debug('Going to pull %d using (%d, %d)', m, u, v)
                auxiliary[u][v]['cap'] -= m
                if auxiliary[u][v]['cap'] == 0:
                    logging.debug('We should remove edge (%d, %d)', u, v)
                    auxiliary[u][v]['used'] = True
                    throughput[v][0] -= m
                    throughput[u][1] += m
                req[v] -= m
                req[u] += m
                q.append(u)
                direction = auxiliary[u][v]['direction']
                if direction == 'B':
                    u, v = v, u
                    m = (-1) * m
                if u not in g:
                    g[u] = {}
                if v not in g[u]:
                    g[u][v] = 0
                g[u][v] += m
                flows.append('(%d, %d) = %d %s' % (u, v, g[u][v], direction))
                logging.debug('Flow (%d, %d) is %d changed by %d direction %s'
                          , u, v, g[u][v], m, direction)
    logging.info('Flows added:\n%s', _to_str(flows))    
     
            
def construct_blocking_flow(source, sink, auxiliary, network, g):
    logging.info('Findig blocking flow')
    while True:
        throughput = calc_throughput(source, sink, auxiliary)
        ret = delete_zero_throughput(source, sink, auxiliary, throughput)
        if not ret:
            logging.debug('Flow is maximal')
            return
        if source not in auxiliary or sink not in auxiliary:
            logging.debug('Flow is maximal')
            return 
        min_thr = (None, sys.maxint)
        for u in throughput:
            current_thr = min(throughput[u][0], throughput[u][1])
            if current_thr < min_thr[1]:
                min_thr = (u, current_thr)
        min_node, min_throughput = min_thr
        logging.debug('Node %d has minimum throughput %d', min_node, 
                      min_throughput)
        push(min_node, min_throughput, auxiliary, throughput, g)
        pull(source, min_node, min_throughput, auxiliary, throughput, g)
    logging.info('Found blocking flow')
    return 


def flow_add(network, g):
    for u, d in g.items():
        v = u
        for node, value in d.items():
            network[v][node]['flow'] += value


def mpm(source, sink, network):
    while True:
        g = {}
        na = build_level_graph(source, sink, network)
        if not na:
            logging.info('done=yes')
            break
        construct_blocking_flow(source, sink, na, network, g)
        flow_add(network, g)
        
    logging.info('Maximum Flow:\n%s',_to_str(network))
    outgoin = [v for v in network[source].iterkeys()]
    maxflow_value = sum([network[source][v]['flow'] for v in outgoin])
    logging.info('Maximum Flow value: %s', str(maxflow_value))
    return network, maxflow_value


def main(fname, source, sink):
    f = open(fname, 'rb')
    logging.info('=====STARTING====')
    network = read_network(f)
    logging.info('Network is loaded')
    mpm(source, sink, network)
    f.close()
    

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        sys.exit('USAGE:\n%s graph_file source sink [loglevel] [logfile]' % sys.argv[0])
    
    loglevel = sys.argv[4] if len(sys.argv) > 4 else 'INFO'
    logfile = sys.argv[5] if len(sys.argv) == 6 else None
    
    logging.basicConfig(format=FORMAT, level=logging.getLevelName(loglevel), 
                        filename=logfile)
    
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    