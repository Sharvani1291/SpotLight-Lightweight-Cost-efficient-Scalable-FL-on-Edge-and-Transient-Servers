#!/usr/bin/env python3.10
import time
import argparse
import os
import sys
import multiprocessing

# user-defined Modules
from Feeder import Feeder
from Orachestrator import Orachestration
from LifeLogger import LifeLogger

class HeartBeat:
    def __init__(self):
        self.start_time = None
        self.current_start_time = None  # Variable to store the start time for the current map
        self.Feeder = Feeder()
        self.orachestrator = Orachestration()
        self.lifeLogger=LifeLogger()

        self.parser = argparse.ArgumentParser(description="Define the configuration for the emulator system")

        # Defines the configuration for the emulator system
        self.parser.add_argument("-az", "--az", type=str, required=True,
                                 help="Defines the az for the emulator system, specify the csv file you want to use")
        self.parser.add_argument("-instance", "--instance", type=str, required=True,
                                 help="Defines the instance type for the system")

    def start(self):
        self.start_time = time.time()

    def runFeeder(self, az, instance):
        # Check if the csv file with the az is in the folder
        csv_file_path = f"{az}.csv"
        if os.path.isfile(csv_file_path):
            print(f"CSV file '{csv_file_path}' exists. Proceeding with calculations.")
            self.Feeder.calculate_lifetime(csv_file_path, instance)
        else:
            print(f"Error: CSV file '{csv_file_path}' does not exist in the folder.")
            sys.exit(1)

    def heart_beat(self, az, instance):
        # Start the clock
        self.start()

        while True:
            # Finds the list of the csv file
            lifetime_files = [filename for filename in os.listdir() if "lifetime" in filename.lower()]

            if lifetime_files:
                # Gets the map of pod and lifetime
                life_time = self.Feeder.map_pod_to_lifetime()

                # Check for matching pods based on their lifetime
                elapsed_time = time.time() - self.start_time
                matching_pods = [pod for pod, lifetime in life_time.items() if elapsed_time >= lifetime]

                if len(matching_pods) == len(life_time):
                    processes = []
                    for pod in matching_pods:
                        # Logs the status of the layer2/3, 1 is for failure and 0 is for success
                        LifeLogger.log_status(1,"layer2")
                        LifeLogger.log_status(1,"layer3")
                        print(f"The following pods have reached their lifetime:")
                        print(pod)
                        
                        #calls the terminate function in the orachestrator
                        self.orachestrator.terminate_pod(pod)
                        time.sleep(3)

                    # Update the start time for the current map
                    self.current_start_time = time.time()
                    self.start_time = self.current_start_time
                    print("The pods have been terminated successfully. Resetting the start time.")
                    LifeLogger.log_status(0,"layer2")
                    LifeLogger.log_status(0,"layer3")

                else:
                    print("No pods have reached their lifetime yet.")

                time.sleep(1)  # Sleep for 1 second between checks

            else:
                self.runFeeder(az, instance)
                print(f"Generated the file for {az}")
                life_time = self.Feeder.map_pod_to_lifetime()

                elapsed_time = time.time() - self.start_time  # Use current_start_time
                matching_pods = [pod for pod, lifetime in life_time.items() if elapsed_time >= lifetime]

                if len(matching_pods) == len(life_time):
                    processes = []
                    for pod in matching_pods:
                        print(f"The following pods have reached their lifetime:")
                        print(pod)
                        process = multiprocessing.Process(target=self.orachestrator.terminate_pod, args=(pod,))
                        processes.append(process)
                        process.start()
                    for process in processes:
                        process.join()

                    # Update the start time for the current map
                    self.current_start_time=time.time()
                    self.start_time = self.current_start_time
                else:
                    print("No pods have reached their lifetime yet.")

                time.sleep(1)  # Sleep for 1 second between checks

if __name__ == "__main__":
    clock = HeartBeat()
    args = clock.parser.parse_args()

    if args.az and args.instance:
        clock.heart_beat(args.az, args.instance)
