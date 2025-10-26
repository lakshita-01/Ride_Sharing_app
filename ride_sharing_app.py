import heapq
import hashlib
import random
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class Graph:
    """GRAPH COMPONENT: Manages routes and navigation using graph data structure"""
    def __init__(self):
        self.vertices = {}
        self.edges = {}
    
    def add_vertex(self, vertex_id, name, x, y):
        """Add a location vertex to the graph"""
        self.vertices[vertex_id] = {
            'name': name,
            'x': x,
            'y': y,
            'neighbors': {}
        }
    
    def add_edge(self, vertex1, vertex2, distance, time):
        """Add a bidirectional edge between vertices"""
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Vertex not found")
        
        self.vertices[vertex1]['neighbors'][vertex2] = {
            'distance': distance,
            'time': time
        }
        self.vertices[vertex2]['neighbors'][vertex1] = {
            'distance': distance,
            'time': time
        }
    
    def dijkstra_shortest_path(self, start, end):
        """Find shortest path using Dijkstra's algorithm"""
        print(f"🔍 GRAPH: Finding shortest path from {start} to {end}")
        
        distances = {vertex: float('inf') for vertex in self.vertices}
        previous = {vertex: None for vertex in self.vertices}
        distances[start] = 0
        
        priority_queue = [(0, start)]
        
        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            if current_vertex == end:
                break
                
            if current_distance > distances[current_vertex]:
                continue
                
            for neighbor, edge_data in self.vertices[current_vertex]['neighbors'].items():
                distance = current_distance + edge_data['distance']
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        
        if distances[end] != float('inf'):
            print(f"✅ GRAPH: Path found - {' → '.join(path)}, Distance: {distances[end]}km")
        else:
            print(f"❌ GRAPH: No path found from {start} to {end}")
        
        return path, distances[end] if distances[end] != float('inf') else None
    
    def get_vertex_coordinates(self, vertex_id):
        """Get coordinates of a vertex"""
        vertex = self.vertices[vertex_id]
        return (vertex['x'], vertex['y'])
    
    def display_graph_info(self):
        """Display graph information"""
        print(f"\n📊 GRAPH INFORMATION:")
        print(f"  Vertices: {len(self.vertices)} locations")
        print(f"  Edges: {sum(len(v['neighbors']) for v in self.vertices.values()) // 2} routes")
        for vertex_id, data in self.vertices.items():
            print(f"    {vertex_id}: {data['name']} - {len(data['neighbors'])} connections")

class PriorityQueue:
    """PRIORITY QUEUE COMPONENT: Manages ride requests with priority based on time and driver availability"""
    def __init__(self):
        self.heap = []
        self.counter = 0  # To handle same priority elements
    
    def add_request(self, request_id, priority, data):
        """Add a ride request to the priority queue"""
        # Lower priority number = higher priority (1 = highest)
        heapq.heappush(self.heap, (priority, self.counter, request_id, data))
        self.counter += 1
        print(f"📥 PRIORITY QUEUE: Added request {request_id} with priority {priority}")
    
    def get_next_request(self):
        """Get the next highest priority request"""
        if not self.heap:
            return None
        priority, counter, request_id, data = heapq.heappop(self.heap)
        print(f"📤 PRIORITY QUEUE: Processing request {request_id} (priority: {priority})")
        return request_id, data
    
    def is_empty(self):
        """Check if queue is empty"""
        return len(self.heap) == 0
    
    def size(self):
        """Get number of requests in queue"""
        return len(self.heap)
    
    def display_queue_status(self):
        """Display current queue status"""
        if self.is_empty():
            print("📭 PRIORITY QUEUE: Empty")
        else:
            # Show requests sorted by priority
            sorted_requests = sorted(self.heap, key=lambda x: x[0])
            print(f"📬 PRIORITY QUEUE: {self.size()} requests")
            for priority, counter, request_id, data in sorted_requests:
                priority_level = "HIGH" if priority == 1 else "MEDIUM" if priority <= 3 else "LOW"
                print(f"    Priority {priority} ({priority_level}): {request_id} - {data['passenger_id']} from {data['pickup_location']}")

class DriverAllocation:
    """HASHING COMPONENT: Driver allocation using hash tables"""
    def __init__(self, initial_drivers=50):
        self.driver_hash_table = {}
        self.available_drivers = set()
        self.driver_locations = {}
        self.driver_ratings = {}
        self.initialize_drivers(initial_drivers)
    
    def _hash_driver_id(self, driver_id, location_hash):
        """Hash function for driver allocation"""
        hash_input = f"{driver_id}_{location_hash}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def initialize_drivers(self, count):
        """Initialize drivers with random locations using hashing"""
        locations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        print(f"🚗 HASHING: Initializing {count} drivers with hash-based allocation")
        
        for i in range(count):
            driver_id = f"DRV_{i:03d}"
            location = random.choice(locations)
            
            location_hash = hashlib.md5(location.encode()).hexdigest()[:4]
            hash_key = self._hash_driver_id(driver_id, location_hash)
            
            self.driver_hash_table[hash_key] = {
                'driver_id': driver_id,
                'location': location,
                'available': True,
                'total_rides': 0,
                'rating': round(random.uniform(4.0, 5.0), 1),
                'earnings': 0.0
            }
            self.available_drivers.add(driver_id)
            self.driver_locations[driver_id] = location
            self.driver_ratings[driver_id] = self.driver_hash_table[hash_key]['rating']
        
        print(f"✅ HASHING: {len(self.driver_hash_table)} drivers added to hash table")
    
    def count_available_drivers_at_location(self, location):
        """Count number of available drivers at a specific location"""
        count = 0
        for hash_key, driver in self.driver_hash_table.items():
            if driver['available'] and driver['location'] == location:
                count += 1
        return count
    
    def count_nearby_available_drivers(self, location, max_distance_locations=2):
        """Count available drivers at nearby locations"""
        # Define nearby locations (simplified - in real app, use actual distance)
        all_locations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        nearby_locations = [location]  # Start with the exact location
        
        # Add some nearby locations (simplified logic)
        location_index = all_locations.index(location)
        nearby_indices = [
            (location_index - 1) % len(all_locations),
            (location_index + 1) % len(all_locations)
        ]
        
        for idx in nearby_indices:
            nearby_locations.append(all_locations[idx])
        
        total_count = 0
        for loc in nearby_locations:
            total_count += self.count_available_drivers_at_location(loc)
        
        return total_count
    
    def find_nearest_driver(self, pickup_location):
        """Find nearest available driver using hashing"""
        print(f"🔍 HASHING: Searching for driver near {pickup_location}")
        
        available_at_location = []
        
        # Hash-based search for drivers at exact location
        for hash_key, driver in self.driver_hash_table.items():
            if (driver['available'] and 
                driver['location'] == pickup_location):
                available_at_location.append(driver)
        
        if available_at_location:
            # Sort by rating (higher rating first)
            available_at_location.sort(key=lambda x: x['rating'], reverse=True)
            print(f"✅ HASHING: Found {len(available_at_location)} drivers at {pickup_location}")
            return available_at_location[0]
        
        # If no drivers at exact location, search nearby using hash table
        print(f"⚠️ HASHING: No drivers at {pickup_location}, searching nearby...")
        for hash_key, driver in self.driver_hash_table.items():
            if driver['available']:
                available_at_location.append(driver)
        
        if available_at_location:
            available_at_location.sort(key=lambda x: x['rating'], reverse=True)
            print(f"✅ HASHING: Found {len(available_at_location)} drivers nearby")
            return available_at_location[0]
        
        print("❌ HASHING: No available drivers found")
        return None
    
    def assign_driver(self, pickup_location, ride_distance):
        """Assign a driver to a ride using hash-based allocation"""
        driver = self.find_nearest_driver(pickup_location)
        
        if driver:
            driver['available'] = False
            self.available_drivers.remove(driver['driver_id'])
            
            # Calculate fare: base + distance rate
            base_fare = 2.50
            distance_rate = 1.75
            fare = base_fare + (ride_distance * distance_rate)
            
            driver['total_rides'] += 1
            driver['earnings'] += fare
            
            print(f"✅ HASHING: Assigned driver {driver['driver_id']} (Rating: {driver['rating']}) - Fare: ${fare:.2f}")
            return driver['driver_id'], fare
        
        return None, 0
    
    def release_driver(self, driver_id, new_location=None):
        """Release driver after ride completion"""
        for hash_key, driver in self.driver_hash_table.items():
            if driver['driver_id'] == driver_id:
                driver['available'] = True
                if new_location:
                    driver['location'] = new_location
                    self.driver_locations[driver_id] = new_location
                self.available_drivers.add(driver_id)
                print(f"🔄 HASHING: Released driver {driver_id} at {new_location}")
                break
    
    def get_driver_stats(self, driver_id):
        """Get statistics for a specific driver"""
        for hash_key, driver in self.driver_hash_table.items():
            if driver['driver_id'] == driver_id:
                return driver.copy()
        return None
    
    def display_hash_table_info(self):
        """Display hash table information"""
        available_count = len(self.available_drivers)
        busy_count = len(self.driver_hash_table) - available_count
        
        # Count drivers per location
        location_counts = defaultdict(int)
        for driver in self.driver_hash_table.values():
            if driver['available']:
                location_counts[driver['location']] += 1
        
        print(f"\n📊 HASH TABLE INFORMATION:")
        print(f"  Total Drivers: {len(self.driver_hash_table)}")
        print(f"  Available Drivers: {available_count}")
        print(f"  Busy Drivers: {busy_count}")
        print(f"  Available Drivers by Location:")
        for location, count in sorted(location_counts.items()):
            print(f"    {location}: {count} drivers")

class RideSharingApp:
    """Main Ride Sharing Application integrating all components"""
    def __init__(self):
        self.graph = Graph()
        self.request_queue = PriorityQueue()
        self.driver_allocator = DriverAllocation()
        self.ride_history = []
        self.request_counter = 0
        self.request_timestamps = {}  # Track request timestamps
        self.priority_counter = 1  # Start with priority 1 (highest)
        
        # Initialize the graph with locations
        self.initialize_graph()
    
    def initialize_graph(self):
        """Initialize the city graph with locations and routes"""
        print("🗺️ INITIALIZING GRAPH COMPONENT...")
        
        # Add vertices (locations in the city)
        locations = [
            ('A', 'Downtown', 0, 0),
            ('B', 'Airport', 10, 8),
            ('C', 'Shopping Mall', 5, 12),
            ('D', 'University', 8, 3),
            ('E', 'Hospital', 12, 6),
            ('F', 'Stadium', 15, 2),
            ('G', 'Park', 3, 7),
            ('H', 'Train Station', 7, 15)
        ]
        
        for vertex_id, name, x, y in locations:
            self.graph.add_vertex(vertex_id, name, x, y)
        
        # Add edges (routes between locations)
        routes = [
            ('A', 'B', 15, 20), ('A', 'D', 8, 12), ('A', 'G', 5, 8),
            ('B', 'E', 6, 10), ('B', 'H', 12, 18),
            ('C', 'G', 8, 15), ('C', 'H', 5, 9),
            ('D', 'E', 7, 11), ('D', 'F', 10, 16),
            ('E', 'F', 6, 9), ('E', 'H', 9, 14),
            ('F', 'H', 13, 20), ('G', 'H', 11, 17)
        ]
        
        for v1, v2, distance, time in routes:
            self.graph.add_edge(v1, v2, distance, time)
        
        self.graph.display_graph_info()
    
    def calculate_priority(self, pickup_location, request_time, current_queue_size):
        """
        Calculate integer priority from 1 to number of rides:
        1 = highest priority, higher numbers = lower priority
        """
        # Base priority starts from 1 and increments
        base_priority = self.priority_counter
        
        # Adjust priority based on driver availability (more drivers = better priority)
        available_drivers = self.driver_allocator.count_nearby_available_drivers(pickup_location)
        
        # If there are many drivers available, give slightly better priority
        if available_drivers >= 3:
            priority_adjustment = -1  # Better priority (lower number)
        elif available_drivers == 0:
            priority_adjustment = 1   # Worse priority (higher number)
        else:
            priority_adjustment = 0   # No adjustment
        
        final_priority = max(1, base_priority + priority_adjustment)
        
        print(f"   🎯 PRIORITY CALCULATION:")
        print(f"     Base priority: {base_priority}")
        print(f"     Available drivers near {pickup_location}: {available_drivers}")
        print(f"     Priority adjustment: {priority_adjustment}")
        print(f"     Final priority: {final_priority}")
        
        return final_priority
    
    def request_ride(self, passenger_id, pickup_location, dropoff_location):
        """Request a new ride with automatic integer priority calculation"""
        self.request_counter += 1
        request_id = f"REQ_{self.request_counter:04d}"
        
        # Get current timestamp for this request
        request_time = datetime.now()
        self.request_timestamps[request_id] = request_time
        
        print(f"\n🎯 NEW RIDE REQUEST {request_id}:")
        print(f"   Passenger: {passenger_id}")
        print(f"   Route: {pickup_location} → {dropoff_location}")
        print(f"   Time: {request_time.strftime('%H:%M:%S')}")
        
        # Calculate route and distance using GRAPH
        path, distance = self.graph.dijkstra_shortest_path(pickup_location, dropoff_location)
        
        if distance is None:
            print(f"❌ RIDE REQUEST FAILED: No route found")
            return None
        
        # Calculate automatic integer priority based on time and driver availability
        current_queue_size = self.request_queue.size()
        priority = self.calculate_priority(pickup_location, request_time, current_queue_size)
        
        # Increment priority counter for next request
        self.priority_counter += 1
        
        request_data = {
            'passenger_id': passenger_id,
            'pickup_location': pickup_location,
            'dropoff_location': dropoff_location,
            'path': path,
            'distance': distance,
            'timestamp': request_time,
            'status': 'pending',
            'calculated_priority': priority
        }
        
        # Add to PRIORITY QUEUE with calculated integer priority
        self.request_queue.add_request(request_id, priority, request_data)
        
        return request_id
    
    def process_ride_requests(self):
        """Process pending ride requests using all components"""
        print(f"\n🔄 PROCESSING RIDE REQUESTS...")
        self.request_queue.display_queue_status()
        self.driver_allocator.display_hash_table_info()
        
        processed_rides = []
        
        while not self.request_queue.is_empty():
            request_id, request_data = self.request_queue.get_next_request()
            
            # Assign driver using HASHING
            driver_id, fare = self.driver_allocator.assign_driver(
                request_data['pickup_location'], 
                request_data['distance']
            )
            
            if driver_id:
                ride_data = {
                    'request_id': request_id,
                    'driver_id': driver_id,
                    **request_data,
                    'fare': fare,
                    'status': 'completed',
                    'completion_time': datetime.now()
                }
                
                # Simulate ride completion and driver release
                self.driver_allocator.release_driver(
                    driver_id, 
                    request_data['dropoff_location']
                )
                
                self.ride_history.append(ride_data)
                processed_rides.append(ride_data)
            
            else:
                # No drivers available
                request_data['status'] = 'failed - no drivers'
                self.ride_history.append({
                    'request_id': request_id,
                    **request_data
                })
                print(f"❌ REQUEST {request_id}: Failed - no drivers available")
        
        print(f"✅ PROCESSING COMPLETE: {len(processed_rides)} rides processed")
        
        # Reset priority counter after processing all requests
        self.priority_counter = 1
        
        return processed_rides
    
    def get_ride_statistics(self):
        """Calculate ride statistics for VISUALIZATION"""
        if not self.ride_history:
            return {}
        
        completed_rides = [ride for ride in self.ride_history if ride.get('status') == 'completed']
        
        stats = {
            'total_rides': len(completed_rides),
            'total_revenue': sum(ride.get('fare', 0) for ride in completed_rides),
            'average_fare': np.mean([ride.get('fare', 0) for ride in completed_rides]) if completed_rides else 0,
            'average_distance': np.mean([ride.get('distance', 0) for ride in completed_rides]) if completed_rides else 0,
            'completion_rate': len(completed_rides) / len(self.ride_history) if self.ride_history else 0,
            'failed_rides': len(self.ride_history) - len(completed_rides)
        }
        
        return stats
    
    def get_driver_statistics(self):
        """Get statistics for all drivers"""
        driver_stats = []
        
        for hash_key, driver in self.driver_allocator.driver_hash_table.items():
            driver_stats.append({
                'driver_id': driver['driver_id'],
                'total_rides': driver['total_rides'],
                'total_earnings': driver['earnings'],
                'rating': driver['rating'],
                'available': driver['available']
            })
        
        return driver_stats

class Visualization:
    """VISUALIZATION COMPONENT: Creates comprehensive ride statistics visualizations"""
    @staticmethod
    def plot_ride_statistics(ride_sharing_app):
        """Plot various ride statistics using matplotlib"""
        print("\n📈 GENERATING VISUALIZATION COMPONENT...")
        
        stats = ride_sharing_app.get_ride_statistics()
        driver_stats = ride_sharing_app.get_driver_statistics()
        
        if not stats['total_rides']:
            print("No completed rides to visualize")
            return
        
        # Create a comprehensive dashboard
        fig = plt.figure("Ride Sharing Analytics Dashboard", figsize=(18, 12))
        
        # Use y parameter instead of pad for title positioning
        fig.suptitle('🚗 Ride Sharing App - Comprehensive Analytics Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Define the grid layout
        gs = plt.GridSpec(3, 4, figure=fig)
        
        # Plot 1: Revenue by Driver (Top 10) - Top-left
        ax1 = fig.add_subplot(gs[0, 0])
        top_drivers = sorted(driver_stats, key=lambda x: x['total_earnings'], reverse=True)[:10]
        driver_ids = [d['driver_id'] for d in top_drivers]
        earnings = [d['total_earnings'] for d in top_drivers]
        
        bars1 = ax1.bar(driver_ids, earnings, color='skyblue', edgecolor='navy', alpha=0.8)
        ax1.set_title('Top 10 Drivers by Earnings\n(HASHING Performance)', fontweight='bold')
        ax1.set_xlabel('Driver ID')
        ax1.set_ylabel('Total Earnings ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, earnings):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'${value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Rides Distribution - Top-middle
        ax2 = fig.add_subplot(gs[0, 1])
        rides_count = [d['total_rides'] for d in driver_stats]
        n, bins, patches = ax2.hist(rides_count, bins=12, color='lightgreen', 
                                   edgecolor='darkgreen', alpha=0.7)
        ax2.set_title('Rides Distribution per Driver\n(PRIORITY QUEUE Impact)', fontweight='bold')
        ax2.set_xlabel('Number of Rides')
        ax2.set_ylabel('Number of Drivers')
        
        # Plot 3: Driver Ratings - Top-right
        ax3 = fig.add_subplot(gs[0, 2])
        ratings = [d['rating'] for d in driver_stats]
        n_ratings, bins_ratings, patches_ratings = ax3.hist(ratings, bins=8, 
                                                           color='gold', edgecolor='orange', alpha=0.7)
        ax3.set_title('Driver Ratings Distribution\n(Service Quality)', fontweight='bold')
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Number of Drivers')
        
        # Plot 4: Overall Statistics - Middle row spanning 3 columns
        ax4 = fig.add_subplot(gs[1, :3])
        categories = ['Total Rides', 'Total Revenue', 'Avg Fare', 'Completion Rate', 'Failed Rides']
        values = [
            stats['total_rides'],
            stats['total_revenue'],
            stats['average_fare'],
            stats['completion_rate'] * 100,
            stats['failed_rides']
        ]
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold', 'lightgray']
        
        bars4 = ax4.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_title('Overall Ride Statistics\n(System Performance Overview)', fontweight='bold')
        ax4.set_ylabel('Values')
        
        # Add value labels on bars
        for bar, value in zip(bars4, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}' if isinstance(value, float) else f'{value}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Component Performance - Bottom row
        ax5 = fig.add_subplot(gs[2, 0])
        components = ['Graph Routing', 'Priority Queue', 'Driver Hashing', 'Visualization']
        efficiency = [95, 88, 92, 90]  # Simulated efficiency scores
        colors_component = ['red', 'blue', 'green', 'purple']
        ax5.bar(components, efficiency, color=colors_component, alpha=0.7)
        ax5.set_title('System Components Efficiency', fontweight='bold')
        ax5.set_ylabel('Efficiency Score (%)')
        ax5.tick_params(axis='x', rotation=15)
        
        # Plot 6: Ride Success vs Failure - Bottom-middle
        ax6 = fig.add_subplot(gs[2, 1])
        success_failure = ['Completed', 'Failed']
        counts = [stats['total_rides'], stats['failed_rides']]
        colors_sf = ['lightgreen', 'lightcoral']
        ax6.pie(counts, labels=success_failure, autopct='%1.1f%%', colors=colors_sf)
        ax6.set_title('Ride Success Rate', fontweight='bold')
        
        # Plot 7: Revenue Trend - Bottom-right
        ax7 = fig.add_subplot(gs[2, 2])
        # Simulate revenue trend
        days = range(1, 8)
        daily_revenue = [stats['total_revenue'] * (0.8 + 0.4 * i/7) for i in days]
        ax7.plot(days, daily_revenue, marker='o', linewidth=2, color='blue')
        ax7.set_title('Weekly Revenue Trend', fontweight='bold')
        ax7.set_xlabel('Day')
        ax7.set_ylabel('Revenue ($)')
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Adjust top margin for title
        plt.show(block=False)
        print("✅ VISUALIZATION: Dashboard generated successfully!")

    @staticmethod
    def plot_city_graph(ride_sharing_app):
        """Plot the city graph with routes"""
        print("\n🗺️ GENERATING GRAPH VISUALIZATION...")
        graph = ride_sharing_app.graph
        
        # Create city map visualization
        fig = plt.figure("City Route Network - Graph Component", figsize=(14, 10))
        
        # Plot vertices
        for vertex_id, vertex_data in graph.vertices.items():
            plt.scatter(vertex_data['x'], vertex_data['y'], s=300, c='red', 
                       zorder=5, alpha=0.8, edgecolors='black')
            plt.text(vertex_data['x'], vertex_data['y'] + 0.5, 
                    f"{vertex_id}\n{vertex_data['name']}", 
                    ha='center', fontsize=9, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Plot edges with different colors based on distance
        for vertex_id, vertex_data in graph.vertices.items():
            for neighbor_id, edge_data in vertex_data['neighbors'].items():
                start = graph.get_vertex_coordinates(vertex_id)
                end = graph.get_vertex_coordinates(neighbor_id)
                
                # Color code by distance
                if edge_data['distance'] < 7:
                    color = 'green'
                    width = 1.5
                elif edge_data['distance'] < 10:
                    color = 'blue'
                    width = 2
                else:
                    color = 'red'
                    width = 2.5
                
                plt.plot([start[0], end[0]], [start[1], end[1]], 
                        color=color, alpha=0.7, linewidth=width, zorder=3)
                
                # Add distance label
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                plt.text(mid_x, mid_y, f"{edge_data['distance']}km", 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        plt.title('City Route Network - Graph Data Structure\n(Shortest Path Routing)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate (km)')
        plt.ylabel('Y Coordinate (km)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add legend for route distances
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=3, label='Short Route (<7km)'),
            plt.Line2D([0], [0], color='blue', lw=3, label='Medium Route (7-10km)'),
            plt.Line2D([0], [0], color='red', lw=3, label='Long Route (>10km)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show(block=False)
        print("✅ GRAPH VISUALIZATION: City map generated successfully!")

# Enhanced Simulation and Demo
def simulate_ride_sharing_app():
    """Run a comprehensive simulation of the ride sharing app"""
    print("=" * 70)
    print("🚗 RIDE SHARING APP SIMULATION - INTEGER PRIORITY SYSTEM")
    print("=" * 70)
    print("🎯 PRIORITY SYSTEM: Integer priorities from 1 to N")
    print("   - 1 = Highest priority, N = Lowest priority")
    print("   - Driver availability adjusts priority within range")
    print("=" * 70)
    
    # Initialize the app
    app = RideSharingApp()
    
    # Generate random ride requests with automatic priority calculation
    locations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    passengers = [f"PASS_{i:03d}" for i in range(1, 16)]  # 15 passengers
    
    print("\n" + "="*50)
    print("📝 PRIORITY QUEUE: Generating Ride Requests")
    print("="*50)
    
    # Create requests with automatic priority calculation
    all_requests = []
    
    # Batch 1: First set of requests (will get priorities 1-5)
    print("\n🕒 BATCH 1: First 5 requests")
    for i in range(5):
        passenger = passengers[i]
        pickup = random.choice(locations)
        dropoff = random.choice([loc for loc in locations if loc != pickup])
        all_requests.append((passenger, pickup, dropoff))
        request_id = app.request_ride(passenger, pickup, dropoff)
        time.sleep(0.05)  # Small delay
    
    # Batch 2: Second set of requests (will get priorities 6-10)
    print("\n🕒 BATCH 2: Next 5 requests")
    for i in range(5, 10):
        passenger = passengers[i]
        pickup = random.choice(locations)
        dropoff = random.choice([loc for loc in locations if loc != pickup])
        all_requests.append((passenger, pickup, dropoff))
        request_id = app.request_ride(passenger, pickup, dropoff)
        time.sleep(0.05)  # Small delay
    
    # Batch 3: Last set of requests (will get priorities 11-15)
    print("\n🕒 BATCH 3: Last 5 requests")
    for i in range(10, 15):
        passenger = passengers[i]
        pickup = random.choice(locations)
        dropoff = random.choice([loc for loc in locations if loc != pickup])
        all_requests.append((passenger, pickup, dropoff))
        request_id = app.request_ride(passenger, pickup, dropoff)
        time.sleep(0.05)  # Small delay
    
    print(f"\n📊 REQUEST SUMMARY:")
    print(f"  Total Requests: {app.request_queue.size()}")
    print(f"  Priority Range: 1 to {app.priority_counter - 1}")
    
    # Process all requests
    print("\n" + "="*50)
    print("🔄 PROCESSING: Executing All Components")
    print("="*50)
    
    completed_rides = app.process_ride_requests()
    
    # Display comprehensive results
    print("\n" + "="*50)
    print("📊 RESULTS: System Performance Analysis")
    print("="*50)
    
    print(f"\n✅ COMPLETED RIDES ({len(completed_rides)}):")
    for ride in completed_rides[:8]:  # Show first 8 completed rides
        priority_level = "HIGH" if ride['calculated_priority'] == 1 else "MEDIUM" if ride['calculated_priority'] <= 3 else "LOW"
        print(f"  ✓ {ride['request_id']}: Priority {ride['calculated_priority']} ({priority_level})")
        print(f"     Passenger: {ride['passenger_id']} | Driver: {ride['driver_id']}")
        print(f"     Route: {ride['pickup_location']}→{ride['dropoff_location']} | ${ride['fare']:.2f} | {ride['distance']}km")
        print()
    
    if len(completed_rides) > 8:
        print(f"  ... and {len(completed_rides) - 8} more rides")
    
    # Display detailed statistics
    stats = app.get_ride_statistics()
    print(f"\n📈 SYSTEM STATISTICS:")
    print(f"  Total Rides Completed: {stats['total_rides']}")
    print(f"  Total Revenue: ${stats['total_revenue']:.2f}")
    print(f"  Average Fare: ${stats['average_fare']:.2f}")
    print(f"  Average Distance: {stats['average_distance']:.2f}km")
    print(f"  Completion Rate: {stats['completion_rate']:.1%}")
    print(f"  Failed Rides: {stats['failed_rides']}")
    
    # Driver statistics
    driver_stats = app.get_driver_statistics()
    top_driver = max(driver_stats, key=lambda x: x['total_earnings'])
    print(f"\n🏆 TOP PERFORMING DRIVER (Hashing Allocation):")
    print(f"  Driver ID: {top_driver['driver_id']}")
    print(f"  Total Rides: {top_driver['total_rides']}")
    print(f"  Total Earnings: ${top_driver['total_earnings']:.2f}")
    print(f"  Rating: {top_driver['rating']}/5.0")
    
    # Component demonstration
    print(f"\n" + "="*50)
    print("🔧 COMPONENT DEMONSTRATION")
    print("="*50)
    
    # Graph component demo
    print(f"\n🗺️ GRAPH COMPONENT DEMO:")
    demo_routes = [('A', 'H'), ('B', 'F'), ('C', 'E')]
    for start, end in demo_routes:
        path, distance = app.graph.dijkstra_shortest_path(start, end)
        if path:
            print(f"  {start} → {end}: {' → '.join(path)} ({distance}km)")
    
    # Priority Queue demo
    print(f"\n📥 PRIORITY QUEUE STATUS:")
    app.request_queue.display_queue_status()
    
    # Hashing demo
    print(f"\n🔑 HASHING COMPONENT STATUS:")
    app.driver_allocator.display_hash_table_info()
    
    # Generate comprehensive visualizations
    print(f"\n" + "="*50)
    print("📊 VISUALIZATION COMPONENT: Generating Analytics")
    print("="*50)
    
    # Enable interactive mode
    plt.ion()
    
    # Generate both visualizations
    Visualization.plot_city_graph(app)
    Visualization.plot_ride_statistics(app)
    
    print(f"\n🎯 SIMULATION COMPLETE!")
    print(f"📊 Two comprehensive dashboards are now displayed:")
    print(f"   1. City Route Network - Graph Component Visualization")
    print(f"   2. Analytics Dashboard - Performance Metrics & Statistics")
    print(f"\n💡 Close all visualization windows to exit the program.")
    
    # Block until all figures are closed
    plt.show(block=True)
    
    return app

if __name__ == "__main__": 
    # Run the enhanced simulation
    app = simulate_ride_sharing_app()