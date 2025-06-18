
import multiprocessing
import os
from kubernetes import client, config
from time import sleep
class Orachestration:

    def __init__(self):
        config.load_kube_config('/etc/rancher/k3s/k3s.yaml')
        self.api_instance = client.CoreV1Api()
    

    #counts the pod no
    def calc_pods(self):

        namespace='default'
        try:
            # Retrieve the list of pods in the namespace
            pod_list = self.api_instance.list_namespaced_pod(namespace)

            # Calculate the number of pods
            num_pods = len(pod_list.items)
            return num_pods
        except Exception as e:
            print(f"Error: {e}")
    

    #return list of pods in the default namespace
    def list_pods(self):
        namespace = 'default'
        try:
            # Retrieve the list of pods in the namespace
            pod_list = self.api_instance.list_namespaced_pod(namespace)

            # Extract pod names
            pod_names = [pod.metadata.name for pod in pod_list.items]

            return pod_names
        except Exception as e:
            print(f"Error: {e}")
    
    #terinate the pods with the given name
    def terminate_pod(self,pod_name):
        namespace="default"
        print("Running terminate pod {}".format(pod_name))
        try:
            self.api_instance.delete_namespaced_pod(name=pod_name,namespace=namespace)
            print(f"Pod {pod_name} terminated successfully.")
        except Exception as e:
            print(f"Error terminating pod {pod_name}: {e}")






