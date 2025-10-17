# Lab 7: Kubernetes Basics - Deployment and NodePort Service

**Goal:** Learn to deploy a basic Nginx web server on a Kubernetes cluster and expose it using a NodePort Service for external access.

### Prerequisites

-   A running Kubernetes cluster (e.g., Minikube, Docker Desktop Kubernetes).
-   `kubectl` command-line tool configured to connect to your cluster.

### Key Kubernetes Concepts

-   **Deployment:** Manages a replicated set of Pods. It ensures that a specified number of Pods are running at any given time.
-   **Pod:** The smallest deployable unit in Kubernetes. A Pod represents a single instance of a running process in your cluster.
-   **Service:** An abstract way to expose an application running on a set of Pods as a network service. Services enable network access to a set of Pods.
    -   **NodePort:** Exposes the Service on each Node's IP at a static port (the NodePort). A NodePort Service routes external traffic directly to your Service. This makes the service accessible from outside the cluster.

### Steps

1.  **Verify Kubernetes Setup**
    Ensure your `kubectl` is configured and connected to a running cluster:
    ```bash
    kubectl cluster-info
    kubectl get nodes
    ```

2.  **Deploy Nginx (Deployment)**
    First, we'll create a Deployment that manages Nginx Pods. This `nginx-deployment.yaml` defines a Deployment named `nginx-deployment` that runs 1 replica of the Nginx image.

    ```bash
    kubectl apply -f nginx-deployment.yaml
    ```
    Verify the Deployment and Pods are running:
    ```bash
    kubectl get deployments
    kubectl get pods
    ```

3.  **Expose Nginx with NodePort Service**
    A NodePort Service exposes the Nginx application on a static port on each Node's IP address. This `nginx-service.yaml` defines a NodePort Service.

    ```bash
    kubectl apply -f nginx-service.yaml
    ```
    Verify the Service:
    ```bash
    kubectl get services
    ```
    To access the service:
    -   If using Minikube: `minikube service nginx-service`
    -   Otherwise: Find your Node's IP address (`kubectl get nodes -o wide`) and the NodePort (`kubectl get services nginx-service`). Then navigate to `http://<NodeIP>:<NodePort>` (the NodePort will be in the range 30000-32767).

4.  **Cleanup**
    To remove all the Kubernetes resources created in this lab:
    ```bash
    kubectl delete deployment nginx-deployment
    kubectl delete service nginx-service
    ```
