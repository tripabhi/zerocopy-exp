# Zero Copy Experiments

This repository contains the code to reproduce our experiments that compare the model load-time for PyTorch's `torch.from_pretrained()`, `torch.load()` and our zero-copy implementation. 

## Prerequisites
You must have docker installed and running in the system. You may verify this, let's say on a Ubuntu machine, using:
```
sudo systemctl status docker
```

## Build Image
Clone this repository using:
```
git clone https://github.com/tripabhi/zerocopy-exp.git
```

Then run:
```
cd zerocopy-exp
docker build --tag zerocopy-exp .
```

You can verify that image is built and available using `docker images`

## Run and test
You can run the Docker container using:
```
docker run -d -p 5000:5000 zerocopy-exp
```

Once the container is up and running, you can test it using
```
curl -X POST -d '{"model":"BERT","input":"<YOUR_TEST_INPUT>"}
```
