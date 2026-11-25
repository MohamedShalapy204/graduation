# Flight Simulation & AI Fault Detection Platform

## System Architecture & Implementation Roadmap

## 1. System Architecture Overview

The system is designed as a **Microservices and Distributed Worker System**, utilizing Docker containers for user-level isolation and resource management. It integrates a high-performance **React frontend**, a **.NET 8/9 Backend & Orchestrator**, a **Dockerized Worker Engine (MATLAB + AI Engine)**, and **pre-computed ANSYS Aerodynamics Lookup Tables**.

### 1.1 High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                               Client Layer (React.js)                             |
|          - 3D Visualizer (Three.js / Cesium.js)                                   |
|          - Telemetry & Sensor Dashboards                                          |
|          - Fault Injection Controls (Bird Strike, Sensor Failures)                |
+-----------------------------------------------------------------------------------+
                                          │
                        WebSockets (WSS) / REST API (HTTPS)
                                          ▼
+-----------------------------------------------------------------------------------+
|                        API Gateway & Core Backend (.NET 8/9)                      |
|          - Authentication & User Sessions                                         |
|          - Docker Orchestration Service (Docker.DotNet)                            |
|          - WebSocket Relay Engine (SignalR Hub)                                   |
+-----------------------------------------------------------------------------------+
                                          │
                            Docker Engine API Socket
                                          ▼
+-----------------------------------------------------------------------------------+
|                         Isolated Worker Containers (Per User)                     |
|  +-----------------------------------------------------------------------------+  |
|  | Container Environment (Linux/Ubuntu)                                       |  |
|  |                                                                             |  |
|  |   +--------------------------+          +-------------------------------+   |  |
|  |   |    AI Fault Detector     |          |  MATLAB Flight Dynamics Engine |   |  |
|  |   |  (Python / ONNX / C++)   |◄────────►|  - Equations of Motion (6-DOF)|   |  |
|  |   +--------------------------+  IPC /   |  - PID/MPC Fault Control      |   |  |
|  |                                Shared   +---------------+---------------+   |  |
|  |                                Memory                  │                   |  |
|  |                                                        ▼                   |  |
|  |                                         +-------------------------------+   |  |
|  |                                         |  ANSYS Lookup Tables (.mat)   |   |  |
|  |                                         |  (Aerodynamics Data Grids)    |   |  |
|  |                                         +-------------------------------+   |  |
|  |                                                                             |  |
|  |   +---------------------------------------------------------------------+   |  |
|  |   | Container Streaming Publisher (gRPC / Internal WebSockets)          |   |  |
|  |   +---------------------------------------------------------------------+   |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Web Frontend (React.js)

* **UI Framework:** React.js + Tailwind CSS.
* **Visualization:** 3D rendering of aircraft posture (Pitch, Roll, Yaw) using `Three.js` or `CesiumJS`. Real-time telemetry plotting using `Chart.js` or `Recharts`.
* **Communication:** `SignalR Client` or native WebSockets for bidirectional low-latency data streams.

### 2.2 Backend & Orchestrator (.NET 8/9)

* **Core API:** RESTful endpoints for authentication, session management, and scenario configuration.
* **Orchestrator:** Interacts directly with `docker.sock` via `Docker.DotNet` library to dynamically spin up, monitor, and destroy worker containers upon user connection or disconnection.
* **SignalR Hub:** Acts as a broker, routing telemetry streams from isolated user containers to their matching React frontend sessions.

### 2.3 Per-User Dynamic Worker Container

* **Isolation:** Each active user scenario runs inside a dedicated, isolated Docker container to prevent state leakage and resource starvation.
* **Environment:** Lightweight Linux base image with MATLAB Runtime / Python runtime.
* **Components:**
  1. **MATLAB Engine:** Runs the 6-DOF (Degrees of Freedom) Flight Dynamics equations.
  2. **AI Fault Detector:** Continuously inspects telemetry for anomalies (e.g., Pitot tube blockages, structural damage).
  3. **ANSYS Pre-Computed Data:** Multi-dimensional lookup tables supplying dynamic Lift ($C_L$), Drag ($C_D$), and Pitch Moment ($C_m$) coefficients under damaged/fault conditions.

---

## 3. Communication Protocols & Data Flow

```
[React Client] ──── HTTP POST /start-scenario ───► [.NET Backend]
                                                         │
                                               Docker API Create & Start
                                                         │
                                                         ▼
                                            [Dynamic Worker Container]
                                                         │
                                                 Simulates Dynamics
                                                         │
[React Client] ◄─── WSS Telemetry (60 FPS) ◄─── [.NET SignalR Broker]
```

### 3.1 Step-by-Step Data Flow

1. **Scenario Initialization:** User selects a scenario on React (e.g., *Bird Strike on Right Wing at* $t = 10s$). React sends HTTP POST to `.NET Backend`.
2. **Container Orchestration:** `.NET` uses `Docker.DotNet` to create a new worker container configured with environment variables: `USER_ID`, `SCENARIO_ID`. Container starts and initializes the MATLAB execution loop.
3. **Simulation Loop & ANSYS Lookup:** At time step $t$, MATLAB evaluates flight state ($\alpha, \beta, V$). If a fault occurs, parameters query the pre-calculated ANSYS grid (`griddedInterpolant` in MATLAB) to update aerodynamic coefficients instantly.
4. **AI Anomaly Detection:** Telemetry values pass through the AI model (ONNX Runtime / C++ API). Upon anomaly detection, AI triggers a `Fault Flag` sent back to MATLAB's control law module. MATLAB modifies elevator/aileron controls to stabilize the aircraft.
5. **Real-time Telemetry Streaming:** Worker Container streams state metrics (JSON / Protobuf) to `.NET SignalR Hub`. SignalR relays data via WebSockets to the client UI at ~30–60 FPS.

---

## 4. AI Fault Detection Model & Dataset Specification

To detect anomalies in real-time within the container, we need an efficient model that runs with minimal latency (< 10ms per inference step).

### 4.1 Recommended AI Models

1. **LSTM-Autoencoder (Long Short-Term Memory):**
   * *Why:* Excellent for time-series anomaly detection. It learns normal flight trajectory patterns and flags high reconstruction error when deviations (sensor failure, structural damage) occur.
2. **Isolation Forest / One-Class SVM (Classical ML):**
   * *Why:* Lightweight, extremely fast inference, highly effective if features are engineered properly from 6-DOF data.
3. **Temporal Convolutional Network (TCN):**
   * *Why:* Faster than RNNs/LSTMs on parallel hardware and captures multi-scale temporal context very well.

### 4.2 Dataset Structure & Features

The model is trained on simulation data generated from clean and fault-injected runs in MATLAB/ANSYS.

* **Input Feature Vector (Time-series window, e.g., last 50 steps):**

  * `Altitude` ($h$)
  * `Airspeed` ($V_{true}$)
  * `Angle of Attack` ($\alpha$)
  * `Sideslip Angle` ($\beta$)
  * `Pitch, Roll, Yaw Angles` ($\theta, \phi, \psi$)
  * `Control Surface Deflections` ($\delta_e, \delta_a, \delta_r$)
  * `Engine Thrust` ($T$)
* **Target Labels (Classification / Detection):**

  * `0`: Normal Flight
  * `1`: Pitot Tube Blockage (Sensor Failure)
  * `2`: Bird Strike (Structural/Aerodynamic Damage)
  * `3`: Gyroscope Drift / Failure
* **Export Format:** Trained models are exported to **ONNX (Open Neural Network Exchange)** format, allowing execution in C++ or Python runtime within the container with near-zero overhead.

---

## 5. Database Schema (PostgreSQL / EF Core)

The database manages users, scenarios, historical simulation logs, and telemetry summaries.

### 5.1 Tables & Relationships

1. **Users Table**

   * `Id` (PK, Guid / int)
   * `Username` (string, Unique)
   * `Email` (string)
   * `PasswordHash` (string)
   * `CreatedAt` (DateTime)
2. **Scenarios Table**

   * `Id` (PK, int)
   * `Name` (string) - e.g., "Severe Bird Strike", "Pitot Failure"
   * `Description` (string)
   * `FaultType` (string) - e.g., `STRUCTURAL_DAMAGE`, `SENSOR_FAILURE`
   * `DefaultParameters` (JSON) - Configuration details for MATLAB/ANSYS lookup.
3. **SimulationSessions Table**

   * `Id` (PK, Guid)
   * `UserId` (FK, Users.Id)
   * `ScenarioId` (FK, Scenarios.Id)
   * `ContainerId` (string) - Docker container ID assigned to this session.
   * `Status` (string) - `Running`, `Completed`, `Failed`, `Terminated`
   * `StartedAt` (DateTime)
   * `EndedAt` (DateTime, Nullable)
4. **SimulationLogs Table** (For post-flight analytics)

   * `Id` (PK, Guid)
   * `SessionId` (FK, SimulationSessions.Id)
   * `Timestamp` (DateTime / Simulation Time ms)
   * `TelemetryData` (JSON) - Snapshot of aircraft state ($V, \alpha, \text{Pitch}$, etc.)
   * `FaultDetected` (boolean)
   * `DetectedFaultType` (string, Nullable)

---

## 6. Detailed API Specifications (.NET Backend)

### 6.1 Authentication APIs

* **`POST /api/auth/register`**
  * *Request Body:* `{ "username": "string", "email": "string", "password": "string" }`
  * *Response (201 Created):* `{ "token": "JWT_STRING", "message": "User registered successfully" }`
* **`POST /api/auth/login`**
  * *Request Body:* `{ "email": "string", "password": "string" }`
  * *Response (200 OK):* `{ "token": "JWT_STRING" }`

### 6.2 Scenario Management APIs

* **`GET /api/scenarios`**
  * *Description:* Get list of available pre-configured flight scenarios (Bird strike, sensor failure, etc.).
  * *Response (200 OK):* List of scenario objects.
* **`GET /api/scenarios/{id}`**
  * *Description:* Get details of a specific scenario.

### 6.3 Simulation Orchestration APIs

* **`POST /api/simulation/start`**
  * *Description:* Requests the backend to provision a Docker container for the user and scenario.
  * *Request Body:* `{ "scenarioId": 1 }`
  * *Response (200 OK):*
    ```json
    {
      "sessionId": "guid-string",
      "containerId": "docker-container-hash",
      "wsEndpoint": "wss://api.flight-sim.com/hub/telemetry"
    }
    ```
* **`POST /api/simulation/stop`**
  * *Description:* Terminates the active session and destroys/stops the user's Docker container.
  * *Request Body:* `{ "sessionId": "guid-string" }`
  * *Response (200 OK):* `{ "message": "Simulation session terminated successfully." }`

### 6.4 Real-time SignalR Hub (`/hub/telemetry`)

* **Client to Server (Invocation):**
  * `SendControlInput`: Sends user inputs (e.g., manual override controls) from React to the container.
* **Server to Client (Broadcast):**
  * `ReceiveTelemetry`: Broadcasts aircraft state, position, and AI fault detection flags at 30–60 FPS.

---

## 7. Implementation Roadmap (Phases & Milestones)

```
+-----------------------------------------------------------------------------------+
| Phase 1: ANSYS Pre-Computation & MATLAB Dynamics Core (Weeks 1 - 3)               |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| Phase 2: AI Model Training & Fault Detection Engine (Weeks 4 - 5)                 |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| Phase 3: Docker Worker Containerization (Weeks 6 - 7)                             |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| Phase 4: .NET Backend & SignalR Streaming Infrastructure (Weeks 8 - 9)            |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| Phase 5: React Frontend & 3D Visualization (Weeks 10 - 11)                        |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| Phase 6: System Integration, Stress Testing & Optimization (Week 12)             |
+-----------------------------------------------------------------------------------+
```

### Phase 1: Aerodynamics & Core Simulation (Weeks 1–3)

* [ ] Run offline ANSYS CFD simulations for clean configuration and damage scenarios (Bird Strike, Surface Damage).
* [ ] Export $C_L, C_D, C_m$ coefficient matrices into MATLAB format (`.mat`).
* [ ] Develop the 6-DOF Flight Dynamics mathematical model in MATLAB.
* [ ] Implement reconfigurable flight control algorithms (PID/MPC) to respond to fault signals.

### Phase 2: AI Anomaly Detection Engine (Weeks 4–5)

* [ ] Generate sensor failure datasets (pitot tube blockage, gyro drift, actuator freeze).
* [ ] Train Anomaly Detection / Classification Model (PyTorch / Scikit-learn).
* [ ] Export trained model to **ONNX format** for fast C++/Python inference within the worker container.

### Phase 3: Worker Containerization (Weeks 6–7)

* [ ] Create Dockerfile containing MATLAB Runtime environment and Python runtime.
* [ ] Build IPC / Internal socket communication between MATLAB execution loop and Python AI runtime.
* [ ] Test container resource limits (CPU quotas, Memory caps).

### Phase 4: Backend & Gateway Development (Weeks 8–9)

* [ ] Build .NET 8/9 Web API with Auth and Session management.
* [ ] Implement Docker SDK integration (`Docker.DotNet`) for automated worker creation/destruction.
* [ ] Build `SignalR Hub` to manage WebSockets channels per user.

### Phase 5: Frontend Development (Weeks 10–11)

* [ ] Build React UI dashboards for scenario controls and metrics display.
* [ ] Integrate `Three.js` / `CesiumJS` for live 3D orientation visualization.
* [ ] Connect SignalR Client for real-time WebSocket telemetry consumption.

### Phase 6: System Integration & Testing (Week 12)

* [ ] End-to-end testing of complete data flow: React $\rightarrow$ .NET $\rightarrow$ Docker Container $\rightarrow$ SignalR $\rightarrow$ React.
* [ ] Stress test multi-user scalability and latency optimization.
* [ ] Final project documentation and presentation prep.
