# Kubernetes Architecture

Kubernetes manages containerized applications
using a control plane and worker nodes.

## Control Plane

The API server exposes the Kubernetes API.
The scheduler selects a suitable node for new pods.
The controller manager maintains the desired state.

## Worker Nodes

Worker nodes run application pods.
The kubelet communicates with the API server.
The container runtime runs the application containers.


## Containers

A container is a lightweight, isolated environment
that runs an application and its dependencies.

Containers package application code, libraries, and
runtime dependencies so applications can run
consistently across environments.

## Pods

A Kubernetes Pod is the smallest deployable unit
in Kubernetes. It can contain one or more containers.

Containers in the same Pod share network resources
and can share storage volumes.


## Troubleshooting

A pod in Pending state may indicate insufficient
resources, scheduling constraints, or storage issues.
A CrashLoopBackOff indicates repeated container failures.
