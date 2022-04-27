import backprop_data

import backprop_network
import numpy as np
import math
import matplotlib.pyplot as plt


def section_b():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    test_accuracy = [None] * 6
    training_accuracy = [None] * 6
    training_loss = [None] * 6

    for i in range(6):
        print("\nLearning rate: " + str(rates[i]) + "\n")
        network = backprop_network.Network([784, 40, 10])
        test_accuracy[i], training_accuracy[i], training_loss[i] = network.SGD(
            training_data,
            epochs=30,
            mini_batch_size=10,
            learning_rate=rates[i],
            test_data=test_data,
        )

    for i in range(6):
        plt.plot(np.arange(30), test_accuracy[i], label="rate = " + str(rates[i]))

    plt.xlabel("epochs")
    plt.ylabel("test accuracy")
    plt.legend()
    plt.show()
    plt.savefig("1.pdf")

    for i in range(6):
        plt.plot(np.arange(30), training_accuracy[i], label="rate = " + str(rates[i]))

    plt.xlabel("epochs")
    plt.ylabel("train accuracy")
    plt.legend()
    plt.show()
    plt.savefig("2.pdf")

    for i in range(6):
        plt.plot(np.arange(30), training_loss[i], label="rate = " + str(rates[i]))

    plt.xlabel("epochs")
    plt.ylabel("train loss")
    plt.legend()
    plt.show()
    plt.savefig("3.pdf")


def section_c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(
        training_data,
        epochs=30,
        mini_batch_size=10,
        learning_rate=0.1,
        test_data=test_data,
    )

section_b()
section_c()
