# Lab 8: Advanced Kubernetes - Services and Persistent Storage

**Goal:** Explore more advanced ways to expose services (ClusterIP, LoadBalancer) and introduce persistent storage using Persistent Volumes (PV) and Persistent Volume Claims (PVC).

### Prerequisites

-   A running Kubernetes cluster (e.g., Minikube, Docker Desktop Kubernetes).
-   `kubectl` command-line tool configured to connect to your cluster.
-   Completion of Lab 7.

### Key Kubernetes Concepts

-   **ClusterIP Service:** Exposes the Service on an internal IP in the cluster. This type makes the Service only reachable from within the cluster, ideal for internal microservice communication.
-   **LoadBalancer Service:** Exposes the Service externally using a cloud provider's load balancer. This is the standard way to expose internet-facing services in a cloud environment.
-   **PersistentVolume (PV):** A piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned. It is a resource in the cluster.
-   **PersistentVolumeClaim (PVC):** A request for storage by a user. It is a request for a PV resource.

### Steps

1.  **Deploy Nginx (Deployment)**
    We will reuse the Nginx Deployment from Lab 7. If you haven't already, apply it:
    ```bash
    kubectl apply -f nginx-deployment.yaml
    ```
    Verify the Deployment and Pods are running:
    ```bash
    kubectl get deployments
    kubectl get pods
    ```

2.  **Expose Nginx with ClusterIP Service**
    A ClusterIP Service makes Nginx accessible only from within the Kubernetes cluster. This `nginx-clusterip-service.yaml` creates a Service that targets the `nginx-deployment`.

    ```bash
    kubectl apply -f nginx-clusterip-service.yaml
    ```
    Verify the Service:
    ```bash
    kubectl get services
    ```
    To access this service from your local machine (for testing purposes), you can use `kubectl port-forward`:
    ```bash
    kubectl port-forward service/nginx-clusterip-service 8080:80
    # Then open http://localhost:8080 in your browser
    ```

3.  **Expose Nginx with LoadBalancer Service**
    A LoadBalancer Service provisions an external load balancer (if your cloud provider supports it) to expose the Nginx application. This `nginx-loadbalancer-service.yaml` defines a LoadBalancer Service.

    ```bash
    kubectl apply -f nginx-loadbalancer-service.yaml
    ```
    Verify the Service and wait for an external IP:
    ```bash
    kubectl get services
    ```
    Once an `EXTERNAL-IP` is assigned, you can access Nginx by navigating to `http://<EXTERNAL-IP>` in your browser.

4.  **Deploy an Application with Persistent Storage (PV & PVC)**
    This section demonstrates how to use Persistent Volumes to store data that persists beyond the life of a Pod. We'll deploy a simple application that writes a timestamp to a file on a mounted volume.

    First, create the Persistent Volume and Persistent Volume Claim:
    ```bash
    kubectl apply -f pv-pvc.yaml
    ```
    Verify they are created and bound:
    ```bash
    kubectl get pv
    kubectl get pvc
    ```
    Now, deploy the application that uses this PVC:
    ```bash
    kubectl apply -f app-with-pv-deployment.yaml
    ```
    Verify the Pod is running:
    ```bash
    kubectl get pods
    ```
    To test persistence, you can exec into the pod and write some data, or use a simple app that does it. For this lab, we'll assume the app writes data. You can then delete and recreate the pod to see the data persist.

5.  **Cleanup**
    To remove all the Kubernetes resources created in this lab:
    ```bash
    kubectl delete deployment nginx-deployment app-with-pv-deployment
    kubectl delete service nginx-clusterip-service nginx-loadbalancer-service
    kubectl delete pvc my-pvc
    kubectl delete pv my-pv
    ```
