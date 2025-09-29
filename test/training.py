import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, file_csv: str):
        self.data_tab = np.genfromtxt(file_csv, delimiter=",", skip_header=1)
        self.x = self.data_tab[:, 0]
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)
        self.y = self.data_tab[:, 1]
        self.m = len(self.x)


class FtLinearRegression:
    def __init__(self, file_csv: str, learning_rate: float, iter_number: int):
        self.__datas = Dataset(file_csv)
        self.__grads = np.array([0.0, 0.0])
        self.__learnnning_rate = learning_rate
        self.__iter_number = iter_number
        self.__tetha = np.array([0.0, 0.0])
        self.__tethas = []
        self.__costs = []

    def model(self, x=None):
        if x is None:
            x = self.__datas.x
        return x * self.__tetha[0] + self.__tetha[1]

    def __cost(self, x):
        error = self.model(x) - self.__datas.y
        loss = (error**2).sum()
        self.__costs.append(loss / (2 * self.__datas.m))

    def __gradients(self, x):
        error = self.model(x) - self.__datas.y
        self.__grads[0] = (error * x).sum() / self.__datas.m
        self.__grads[1] = error.sum() / self.__datas.m

    def __gradient_descent(self, x):
        for _ in range(self.__iter_number):
            self.__gradients(x)
            self.__tetha = self.__tetha - self.__learnnning_rate * self.__grads
            self.__tethas.append(self.__tetha)
            self.__cost(x)

    def get_result(self):
        return self.__datas.x, self.__datas.y, self.__tetha, self.__tethas, self.__costs

    def standardisation(self):
        return (self.__datas.x - self.__datas.x_mean) / self.__datas.x_std

    def destandardisation(self):
        tetha0 = self.__tetha[0] / self.__datas.x_std
        tetha1 = self.__tetha[1] - (
            self.__tetha[0] * self.__datas.x_mean / self.__datas.x_std
        )
        self.__tetha = np.array([tetha0, tetha1])

    def train(self):
        x_std = self.standardisation()
        self.__gradient_descent(x_std)
        self.destandardisation()


def main():
    linear_regression = FtLinearRegression("data.csv", 0.005, 1500)
    linear_regression.train()
    x, y, tetha, tethas, costs = linear_regression.get_result()
    plt.figure()
    plt.plot(x, y, "o")
    plt.plot(x, linear_regression.model())
    plt.show()

    plt.figure()
    plt.plot(tethas)
    plt.show()

    plt.figure()
    plt.plot(costs)
    plt.show()

    print("theta ", tetha)
    return 0


if __name__ == "__main__":
    main()
