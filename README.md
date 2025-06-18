# Scale-FL

# Running the RAFT Nodes

1. python3 run_raft.py -i 2 -a 127.0.0.1:5020 -e 1/127.0.0.1:5010,3/127.0.0.1:5030
2. python3 -m pyraft.run_raft -i 1 -a 127.0.0.1:5010
3. python3 run_raft.py -i 3 -a 127.0.0.1:5030 -e 1/127.0.0.1:5010,2/127.0.0.1:5020
