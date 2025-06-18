#picks the lifetime of the system
import pandas as pd
import csv
import os
import sys
import numpy as np
#userDefined Modules
from Orachestrator import Orachestration
from sklearn.preprocessing import MinMaxScaler
from time import sleep
class Feeder:
    def __init__(self):
        self.Orachestrator=Orachestration()
       
    ##methods reads the dataset into the df
    def read(self,az,instance):
        df = pd.read_csv(az)

        filtered_df = df[df['Instance Type'] == instance]
        filtered_df = filtered_df[["Price", "Timestamp"]]

        # Sort the DataFrame by timestamp
        filtered_df = filtered_df.sort_values(by='Timestamp')

        # Create a new column 'VMid' with ascending values
        filtered_df['VMid'] = range(1, len(filtered_df) + 1)

        return filtered_df.set_index('VMid')

    
    def calculate_lifetime(self, az, instance):
        filtered_df = self.read(az, instance)

        bid_strategy = 1.0
        VMs = []
        VMIDs = []
        max_bids = 1

        for i in range(max_bids):
            bid = bid_strategy
            VMs.append(i)

            VMs[i] = dict()
            vmid = 0
            cp_sorted_data = filtered_df.copy()

            for index, row in cp_sorted_data.iterrows():
                utime = row['Timestamp']
                price = row['Price']  # Include price information

                if len(VMIDs) == 0:
                    VMIDs.append(vmid)
                elif vmid > VMIDs[-1]:
                    VMIDs.append(vmid)

                if vmid not in VMs[i]:
                    VMs[i][vmid] = dict()
                    VMs[i][vmid]["price"] = price * bid
                    VMs[i][vmid]["start"] = utime
                    VMs[i][vmid]["end"] = None
                    VMs[i][vmid]["duration"] = None
                else:
                    if price > VMs[i][vmid]["price"]:
                        VMs[i][vmid]["end"] = utime
                        VMs[i][vmid]["duration"] = VMs[i][vmid]["end"] - VMs[i][vmid]["start"]

                        if VMs[i][vmid]["start"] > VMs[i][vmid]["end"]:
                            print("Error start time is higher than end", i, vmid, VMs[i][vmid])
                            return pd.DataFrame()

                cp_sorted_data = cp_sorted_data.iloc[1:]

                for next_index, next_row in cp_sorted_data.iterrows():
                    next_time = next_row['Timestamp']
                    next_price = next_row['Price']

                    if next_price <= VMs[i][vmid]["price"]:
                        continue
                    else:
                        VMs[i][vmid]["end"] = next_time
                        VMs[i][vmid]["duration"] = VMs[i][vmid]["end"] - VMs[i][vmid]["start"]
                        break

                vmid += 1

        output_file = az + "-lifetime.csv"
        with open(output_file, 'w', newline='') as f:
            header = ['VMID', 'Duration', 'Timestamp', 'Price']  # Include 'Price' in the header
            writer = csv.writer(f)
            writer.writerow(header)

            for vmid in VMIDs:
                if VMs[0][vmid]["duration"] is not None:
                    row = [vmid, int(VMs[0][vmid]["duration"]), int(VMs[0][vmid]["start"]), VMs[0][vmid]["price"]]
                    writer.writerow(row)

        print("Results saved to", output_file)
        return "The results saved to", output_file


    #picks the lifetime of the system
    def choose_lifetime(self):

        #duration is the number of seconds
        duration_min=60
        duration_max=2400
        
        #choosing random seed
        np.random.seed(42)
        #finds the list of the csv file
        lifetime_files = [filename for filename in os.listdir() if "lifetime" in filename.lower()]
        if lifetime_files:

            #find the values to be returned
            num_pods=self.Orachestrator.calc_pods()
            lifetime_file = lifetime_files[0]
            df_lifetime=pd.read_csv(lifetime_file)

            #scaling the values between 2 min and 40 mins
            duration_column = df_lifetime['Duration'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(duration_min,duration_max))
            df_lifetime["Duration"]=scaler.fit_transform(duration_column)

            #rounding the values to integers

            df_lifetime["Duration"]=round(df_lifetime["Duration"]).astype(int)
            
            #chooses the lifetime of the pods
            random_values = df_lifetime["Duration"].sample(n=num_pods).to_list()

            #chooses the price of the pods
            #random_prices=df_lifetime["Price"].sample(n=num_pods).to_list()

            # Display or return the randomly picked values
            # print("Randomly picked values:")
            # print(random_values)
            return random_values

        else:
            print("No 'lifetime' files found in the folder.")
            sys.exit(1)
    
    #Mapping pods to life
    def map_pod_to_lifetime(self):
        pod_list=self.Orachestrator.list_pods()
        lifetime_list=self.choose_lifetime()
        if len(pod_list) != len(lifetime_list):
            print("Error: Number of pods and lifetime values do not match.")
            print("Re-calculating the number of pods in the system")
            sleep(10)
            pod_list=self.Orachestrator.list_pods()
            

        pod_lifetime_mapping = dict(zip(pod_list, lifetime_list))
        print(pod_lifetime_mapping)
        return pod_lifetime_mapping



    



   

