# SpotLight: Lightweight, Cost-Efficient, and Scalable Federated Learning on Edge and Transient Servers

**SpotLight** is a federated learning framework designed for efficient, scalable, and fault-tolerant deployment across edge and transient cloud servers. It emphasizes cost-efficiency, asynchronous operation, and dynamic resource utilization.

---

## ✨ Key Features

- ⚡ **Lightweight Clients**: Designed for edge devices with limited resources.
- 🔁 **Asynchronous Updates**: Avoids stragglers and reduces client idle time.
- 🧠 **Client-Side Selection**: Clients self-assess eligibility based on local conditions, reducing server load.
- 💸 **Cost-Efficient Cloud Execution**: Supports transient/preemptible cloud instances.
- 📈 **Resilient Aggregation**: Aggregator tracks client round metadata for consistent model convergence despite client variability.
- 🔍 **Pluggable Models**: Easily extendable to different deep learning models.

---

## 🏗️ Architecture Overview

SpotLight is composed of:
- **Client Code**: Responsible for training, model updates, and eligibility logic.
- **Server Code**: Coordinates aggregation and round management.
- **GRPC Communication**: Lightweight RPC between clients and server.
- **Client Selection Modules**: Adaptive mechanisms for client eligibility.
- **Load Generator**: Simulates clients to test scalability and stability.

---

## 📁 Repository Structure

| Folder / File                 | Description |
|------------------------------|-------------|
| `Client_Code/`               | Federated learning client logic |
| `Server_Code/`               | Server-side aggregation and coordination |
| `ClientSelection_clientside/`| Client-side selection policy logic |
| `ClientSelection_serverside/`| Server-side optional selection strategies |
| `GRPC-*` / `Emulator/`       | gRPC-based networking and simulation components |
| `load-generator/`            | Utility to simulate high-volume client activity |
| `Raft/`                      | Distributed consensus mechanisms (optional) |
| `models.py`                  | Model definitions for training |
| `haproxy.cfg`                | HAProxy configuration for testing load balancing |
| `README.md`                  | You're reading it :) |

---

## 🚀 How to Run

### Build gRPC modules:
```bash
make
```

### Run server:
```bash
python3 GRPC-server/server.py
```

### Launch clients (can be parallelized):
```bash
python3 GRPC-client/client.py --id CLIENT_ID
```

You can use the load-generator to simulate large-scale clients.

---

## 📊 Evaluation

SpotLight has been evaluated on multiple settings:
- Realistic edge-client churn and delay
- Varying network latencies
- Comparisons with FedAvg, FedLite, and asynchronous FL methods
- Scenarios with cloud preemption and client dropout

---

## 📚 References

SpotLight draws on ideas from:
- Papaya
- FLWR
- Totoro
- FedScale
- FedLite
- FedSEA
- Adaptive Federated Optimization (FedAdam, FedYogi, FedAdagrad)
- Asynchronous Federated Optimization
- FATE-LLM

---

## 👩‍💻 Author

Sharvani Chelumalla  
M.S. in Computer Science – University of Georgia  

---

## 📜 License

This project is for academic research and educational purposes only.
