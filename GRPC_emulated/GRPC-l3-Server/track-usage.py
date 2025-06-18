import psutil
import time
import csv
from datetime import datetime

def log_resource_usage():
    with open('resource_usage_l3.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Get the number of CPU cores
        num_cores = psutil.cpu_count(logical=True)
        # Create header with CPU usage per core
        header = ['Timestamp', 'RAM Usage (%)', 'Disk Read (bytes)', 'Disk Write (bytes)', 'Network Sent (bytes)', 'Network Received (bytes)']
        header.extend([f'CPU Core {i} Usage (%)' for i in range(num_cores)])
        writer.writerow(header)
        
        while True:
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get CPU usage per core
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            # Get RAM usage
            ram_usage = psutil.virtual_memory().percent
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes
            disk_write = disk_io.write_bytes
            
            # Get network usage
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
            
            # Write the resource usage to the CSV file
            row = [timestamp, ram_usage, disk_read, disk_write, net_sent, net_recv]
            row.extend(cpu_per_core)
            writer.writerow(row)
            file.flush()
            
            # Sleep for a specified interval before logging again
            time.sleep(5)

if __name__ == '__main__':
    log_resource_usage()