# 🛰️ Garuda Drishti — Autonomous Satellite Fault Detection System
Garuda Drishti (meaning Eagle's Eye in Sanskrit) is an AI-powered satellite health monitoring system that automatically detects anomalies in satellite telemetry data using deep learning.

The system mimics how real space agencies monitor satellites in orbit, where manual intervention is impossible and faults must be detected autonomously before they lead to mission failure.

🚀 Project Overview

Satellites operate in extremely harsh environments in space. When a sensor fails or behaves abnormally, it can lead to severe system malfunctions.

Traditional rule-based monitoring systems struggle to detect complex patterns across multiple sensor channels.

Garuda Drishti solves this problem using an LSTM Autoencoder model that learns normal satellite behavior and identifies abnormal patterns automatically.

⚠️ Problem Statement

In space missions:

Manual repair is impossible

Sensor failures can cause mission loss

Telemetry data is complex and multi-dimensional

Therefore, an intelligent autonomous fault detection system is required to monitor satellite health in real time.

⚙️ How the System Works
1️⃣ Telemetry Data Simulation

The system generates realistic satellite telemetry data:

7000 timesteps

25 sensor channels

These simulate parameters such as:

Thermal variations

Power system fluctuations

Attitude control signals

Fuel levels

Environmental variations

The dataset mimics patterns similar to telemetry from NASA SMAP and ISRO satellite missions.

2️⃣ LSTM Autoencoder Model

The AI model is trained only on normal satellite behavior.

Architecture:

Encoder → compresses sensor data
Decoder → reconstructs the original data

Key idea:

Normal data → reconstructed accurately

Faulty data → reconstructed poorly

This difference allows the model to detect anomalies.

3️⃣ Anomaly Detection

The system calculates reconstruction error between:

Original data vs Reconstructed data

Threshold formula:

Threshold = Mean Error + 2 × Standard Deviation

If the error exceeds the threshold:

🚨 Fault Detected
