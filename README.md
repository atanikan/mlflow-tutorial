# MLflow

Presentation can be found here: https://docs.google.com/presentation/d/1NOuwkOMBxKz_CePfYCeVZ2zhq-MDPcQqCOKGzf9N06E/edit?usp=sharing

## Tracking server
To run a tracking server which also provides a database, you can use the docker container provided [here](https://hub.docker.com/r/atcommons/mlflow-server)  

To build & run the singularity container on Theta GPU computes or Theta is given below

```bash
singularity build mlflow-server.sif docker://atcommons/mlflow-server
singularity exec --bind /home/atanikanti/mlflow-tutorial/artifacts:/mlflow mlflow-server.sif mlflow server --port 8080 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /home/atanikanti/mlflow-tutorial/artifacts
```

To access the server use port forwarding

```bash
ssh -L 8080:127.0.0.1:8080 <username>@theta.alcf.anl.gov

```



## MLflow Project

Aids in packaging multi step code and sharing with others to reproduce
mlflow run

```bash
mlflow run https://github.com/atanikan/mlflow-tutorial
```

## MLflow Models




