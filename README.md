# 🚗 Ride Sharing Application
A comprehensive ride-sharing simulation application demonstrating core data structures and algorithms including Graph-based routing, Priority Queue for ride requests, Hashing for driver allocation, and advanced visualization.

# 📋 Table of Contents
Overview

Features

System Architecture

Installation

Usage

Data Structures Used

Algorithm Implementation

Project Structure

Visualization

Sample Output

Future Enhancements

Contributing

License

# 🌟 Overview
This project simulates a complete ride-sharing ecosystem with intelligent driver allocation, optimal route planning, and real-time analytics. The application showcases how different data structures work together to solve complex real-world problems.

# ✨ Features
# 🗺️ Graph-Based Routing
Shortest Path Calculation using Dijkstra's Algorithm

City Map Visualization with interactive routes

Multiple Location Support with real coordinates

Optimal Route Planning between any two points

# 📥 Priority Queue System
Integer Priority Assignment (1 = Highest, N = Lowest)

Time-based Prioritization with driver availability tie-breaking

Dynamic Priority Adjustment based on real-time conditions

Efficient Request Processing using heapq

# 🔑 Hashing for Driver Allocation
Hash-based Driver Lookup for optimal performance

Location-based Allocation using MD5 hashing

Driver Rating System for quality matching

Real-time Availability Management

# 📊 Advanced Visualization
Interactive City Map with route visualization

Comprehensive Analytics Dashboard

Real-time Statistics and performance metrics

Driver Performance Tracking

# 🏗️ System Architecture
text
Ride Sharing App
├── Graph Component (Route Planning)
├── Priority Queue (Request Management)
├── Hashing System (Driver Allocation)
└── Visualization Engine (Analytics)
# 🚀 Installation
Prerequisites
Python 3.8+
Required packages: matplotlib, numpy
Setup
bash
# Clone the repository
git clone https://github.com/lakshita-01/Ride_Sharing_app
cd ride-sharing-app

# Install required packages
pip install matplotlib numpy
Required Dependencies
bash
pip install matplotlib==3.5.0
pip install numpy==1.21.0
# 💻 Usage
Running the Application
bash
python ride_sharing_app.py
Code Structure
python
# Main components
app = RideSharingApp()                    # Initialize application
app.request_ride(passenger, pickup, dropoff)  # Request a ride
app.process_ride_requests()               # Process all pending requests
Example Usage
python
# Initialize the ride sharing system
app = RideSharingApp()

# Request rides with automatic priority calculation
app.request_ride("PASS_001", "A", "H")  # From Downtown to Train Station
app.request_ride("PASS_002", "B", "D")  # From Airport to University

# Process all ride requests
completed_rides = app.process_ride_requests()

# View statistics
stats = app.get_ride_statistics()
print(f"Completed Rides: {stats['total_rides']}")
print(f"Total Revenue: ${stats['total_revenue']:.2f}")
🏗️ Data Structures Used
# 1. Graph Data Structure
Purpose: Route planning and navigation

Implementation: Adjacency list with vertex and edge management

Operations: Add vertex, add edge, shortest path calculation

# 2. Priority Queue
Purpose: Manage ride requests with intelligent prioritization

Implementation: Min-heap using heapq

Priority System: Integer priorities 1-N with time and driver availability factors

# 3. Hash Table
Purpose: Efficient driver allocation and management

Implementation: MD5-based hashing with collision handling

Features: Location-based lookup, availability tracking

# 4. Supporting Structures
Dictionaries: For vertex storage and driver management

Sets: For tracking available drivers

Lists: For ride history and statistics

# ⚙️ Algorithm Implementation
# Dijkstra's Algorithm
python
def dijkstra_shortest_path(self, start, end):
    # Implementation of shortest path finding
    # Time Complexity: O((V + E) log V)
# Priority Queue Operations
python
def add_request(self, request_id, priority, data):
    # O(log n) insertion using heapq
Driver Allocation
python
def find_nearest_driver(self, pickup_location):
    # O(1) average case lookup using hashing
# 📁 Project Structure
text
ride_sharing_app/
├── ride_sharing_app.py          # Main application file
├── classes/
│   ├── Graph.py                 # Graph data structure
│   ├── PriorityQueue.py         # Priority queue implementation
│   ├── DriverAllocation.py      # Hashing-based driver management
│   └── Visualization.py         # Plotting and analytics
├── data/
│   ├── city_locations.json      # City map data
│   └── driver_data.json         # Driver information
└── tests/
    ├── test_graph.py            # Graph functionality tests
    ├── test_priority_queue.py   # Queue operations tests
    └── test_driver_allocation.py # Driver management tests
# 📊 Visualization
The application generates two comprehensive visualizations:

# 1. City Route Network
Interactive city map with all locations

Color-coded routes based on distance

Real-time path highlighting

Distance labels on all routes

# 2. Analytics Dashboard
Top drivers by earnings

Rides distribution per driver

Driver ratings distribution

Overall system statistics

Component efficiency metrics

Ride success rate pie chart

Revenue trends

📈 Sample Output
text
🚗 RIDE SHARING APP SIMULATION - INTEGER PRIORITY SYSTEM
==================================================
🎯 PRIORITY SYSTEM: Integer priorities from 1 to N
   - 1 = Highest priority, N = Lowest priority
   - Driver availability adjusts priority within range
==================================================

📝 PRIORITY QUEUE: Generating Ride Requests
==================================================

🎯 NEW RIDE REQUEST REQ_0001:
   Passenger: PASS_001
   Route: A → H
   Time: 14:30:25
🔍 GRAPH: Finding shortest path from A to H
✅ GRAPH: Path found - A → B → H, Distance: 27km
   🎯 PRIORITY CALCULATION:
     Base priority: 1
     Available drivers near A: 4
     Priority adjustment: -1
     Final priority: 1
📥 PRIORITY QUEUE: Added request REQ_0001 with priority 1

✅ PROCESSING COMPLETE: 15 rides processed

📊 RESULTS: System Performance Analysis
==================================================
✅ COMPLETED RIDES (12):
  ✓ REQ_0001: Priority 1 (HIGH)
     Passenger: PASS_001 | Driver: DRV_015
     Route: A→H | $49.75 | 27km

📈 SYSTEM STATISTICS:
  Total Rides Completed: 12
  Total Revenue: $324.50
  Average Fare: $27.04
  Average Distance: 14.08km
  Completion Rate: 80.0%
  Failed Rides: 3
🔧 Configuration
City Locations
The application comes pre-configured with 8 locations:

A: Downtown (0, 0)

B: Airport (10, 8)

C: Shopping Mall (5, 12)

D: University (8, 3)

E: Hospital (12, 6)

F: Stadium (15, 2)

G: Park (3, 7)

H: Train Station (7, 15)

Driver Configuration
Initial Drivers: 50

Rating Range: 4.0 - 5.0

Location Distribution: Random across all locations

Pricing Model
Base Fare: $2.50

Distance Rate: $1.75 per km

Calculation: fare = base_fare + (distance * distance_rate)

# 🚀 Future Enhancements
Planned Features
Real-time GPS tracking integration

Machine learning for demand prediction

Multi-vehicle type support (Economy, Premium, SUV)

Dynamic pricing based on demand

Mobile application interface

Database persistence

User authentication system

Payment gateway integration

Ride scheduling feature

Driver performance analytics

# Technical Improvements
Database integration (PostgreSQL)

REST API development

WebSocket for real-time updates

Docker containerization

Performance optimization

Enhanced error handling

Comprehensive test suite

CI/CD pipeline

# 🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Contribution Guidelines
Follow PEP 8 style guide

Write comprehensive tests

Update documentation

Ensure backward compatibility


# 🎓 Academic Use
This project is ideal for:

Data Structures and Algorithms courses

Software Engineering projects

System Design demonstrations

Academic Research in transportation systems

# Learning Objectives
Understand graph algorithms in real-world applications

Learn priority queue implementation and usage

Explore hashing techniques for efficient data retrieval

Practice object-oriented programming principles

Develop data visualization skills

# 🙏 Acknowledgments
Dijkstra's Algorithm for shortest path finding

Python heapq module for priority queue implementation

Matplotlib for data visualization

NumPy for statistical calculations

⭐ Star this repo if you find it helpful!

🐛 Found a bug? Please open an issue with detailed description.

💡 Have an idea? We'd love to hear your suggestions!

📥 Download Instructions
To download and use this project:

Download the complete code as a ZIP file

Extract the contents to your desired directory

Install dependencies as mentioned in the Installation section

Run python ride_sharing_app.py to start the simulation

# Quick Start for Educators:
bash
# Download and run in one command (Linux/Mac)
wget https://github.com/yourusername/ride-sharing-app/archive/main.zip
unzip main.zip
cd ride-sharing-app-main
pip install matplotlib numpy
python ride_sharing_app.py
# For Students:
Perfect for data structures and algorithms projects

Comprehensive documentation for understanding

Easy to modify and extend

Well-commented code for learning

# File Structure Download:
text
ride-sharing-app/
├── README.md (this file)
├── ride_sharing_app.py (main application)

# Note: Make sure you have Python installed on your system before running the application.
