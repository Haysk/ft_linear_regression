import numpy as np
from training import FtLinearRegression


def get_saved_tetha(file_name):
    try:
        theta =  np.genfromtxt(file_name, delimiter=",", dtype=float)
    except FileNotFoundError as e:
        return np.array([0.0, 0.0])
    if np.isnan(theta).any() or theta.shape[0] != 2:
        raise Exception(f"Le fichier {file_name} est mal formate")
    return theta


def main():
    result = None
    try:
        theta = get_saved_tetha("tetha.csv")
        lr = FtLinearRegression()
        lr.set_tetha(theta)
        while result is None:
            x = input("Entrez le nombre de Km (entre 0 - 390000) pour connaitre le prix de votre voiture :\n")
            try:
                if x == "":
                    raise Exception("\nLe nombre de Km ne doit pas etre vide\n")
                x = int(x)
                if x < 0 or x > 390000:
                    raise Exception("\nLe nombre de Km doit être entre 0 et 390000\n")
                result = lr.model(x)
                result = round(result, 2)
                print(f"Le prix de votre voiture est de {result}")
            except ValueError:
                print("\nLe nombre de Km doit être un entier\n")
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
