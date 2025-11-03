import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, file_csv: str):
        if file_csv == "":
            self.data_tab = None
        else:
            self.data_tab = np.genfromtxt(
                file_csv, delimiter=",", skip_header=1, dtype=float
            )
            if np.isnan(self.data_tab).any():
                print("Le fichier n'est pas un fichier csv valide")
                raise ValueError
            self.x = self.data_tab[:, 0]
            self.x_mean = np.mean(self.x)
            self.x_std = np.std(self.x)
            self.y = self.data_tab[:, 1]
            self.m = len(self.x)

class FtLinearRegression:
    def __init__(self, file_csv: str = ""):
        self.__datas = Dataset(file_csv)
        self.__grads = np.array([0.0, 0.0])
        self.__tetha = np.array([0.0, 0.0])
        self.__tethas = []
        self.__costs = []

    def __cost(self, x):
        error = self.model(x) - self.__datas.y
        loss = (error**2).sum()
        self.__costs.append(loss / (2 * self.__datas.m))

    def __gradients(self, x):
        error = self.model(x) - self.__datas.y
        self.__grads[0] = (error * x).sum() / self.__datas.m
        self.__grads[1] = error.sum() / self.__datas.m

    def __gradient_descent(self, x, learning_rate, iter_number):
        for _ in range(iter_number):
            self.__gradients(x)
            tetha = self.__tetha - learning_rate * self.__grads
            self.__tetha = tetha
            self.__tethas.append(self.__tetha)
            self.__cost(x)

    def __standardisation(self):
        return (self.__datas.x - self.__datas.x_mean) / self.__datas.x_std

    def __destandardisation(self):
        tetha0 = self.__tetha[0] / self.__datas.x_std
        tetha1 = self.__tetha[1] - (
            self.__tetha[0] * self.__datas.x_mean / self.__datas.x_std
        )
        self.__tetha = np.array([tetha0, tetha1])

    def model(self, x=None):
        if x is None:
            if self.__datas.data_tab is None:
                return 0
            x = self.__datas.x
        return x * self.__tetha[0] + self.__tetha[1]

    def set_tetha(self, tetha):
        self.__tetha = tetha

    def save_tetha(self, file_name):
        np.savetxt(file_name, self.__tetha, delimiter=",")

    def train(self, learning_rate, iter_number):
        if self.__datas.data_tab is None:
            raise Exception("No DataSet")
        x_std = self.__standardisation()
        self.__gradient_descent(x_std, learning_rate, iter_number)
        self.__destandardisation()

    def display(self, x_label, y_label):
        if self.__datas.data_tab is None:
            raise Exception("No DataSet")
        figure = plt.figure(figsize=(6, 4))
        figure.set_figwidth(15)
        figure.set_figheight(15)
        ax0 = figure.add_subplot(2, 1, 1)
        ax0.plot(self.__datas.x, self.__datas.y, "o")
        ax0.plot(self.__datas.x, self.model())
        ax0.set_xlabel(x_label)
        ax0.set_ylabel(y_label)
        ax0.set_title("Regression Lineaire")

        ax1 = figure.add_subplot(2, 2, 3)
        ax1.plot(self.__tethas)
        ax1.set_title("evolution des tethas")

        ax2 = figure.add_subplot(2, 2, 4)
        ax2.plot(self.__costs)
        ax2.set_title("evolution du cout")
        plt.show()


def main():
    try:
        linear_regression = FtLinearRegression("data.csv")
        linear_regression.train(0.005, 1500)
        linear_regression.display("Km des voitures", "Prix des voitures")
        linear_regression.save_tetha("tetha.csv")
    except FileNotFoundError as e:
        print(f"le fichier {str(e)[:-11]} n'existe pas")
    except Exception as e:
        print(e)
    return 0


if __name__ == "__main__":
    main()
