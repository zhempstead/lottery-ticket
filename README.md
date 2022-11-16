# lottery-ticket
Exploring lottery ticket hypothesis

# Setup
```
mkdir model
mkdir plots
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

# Run the example script
```
source venv/bin/activate
python src/train_feedforward_mnist.py
```

# Notes
- I haven't yet added dropout or other regularization to the network but that should be trivial
- I used ReLU instead of tanh as that is the norm nowadays
- Pruning the final layer at only half the rate makes a big difference!
- Pruning before training makes the model train much worse, as expected
- Iterative pruning should be easy to do with the callback argument I added to the training step.
- The paper uses 50 epochs; my example script uses only 10
- Takes only a few minutes to train on a relatively beefy machine with ~48 CPUs but not GPU. I think there are a couple of lines of code that would be needed to make it run on GPUs.
