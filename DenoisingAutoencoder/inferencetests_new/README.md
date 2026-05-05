These all store inference tests for various models that were supposed to be upgrades to
the one trained for 5 epochs on 70 images. The highest performing one seems to be the one
trained for 50 epochs on 138 images, so I'm currently training another for 100 epochs. 
For now, though, use the results found in inferencetest_50e. Each folder also contains
the saved model so you can load it yourself, but I saved the models using a newer version
of keras (since I didn't want to try and download conda on the HPC cluster), so you probably
can't load them in this environment.