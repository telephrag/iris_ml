
import random

# This file contains script generating data about Iris flower and appending it to file

def generate(kind, smin, smax, pmin, pmax, path, count=100, rseed=228):
    with open(path, 'a') as f:
        f.write("\n")

        r = random.Random(rseed)
        for _ in range(count):
            sepals = round(r.uniform(smin, smax), 1)
            petal  = round(r.uniform(pmin, pmax), 1)
            f.write(f"{sepals},0.0,{petal},0.0,{kind}\n")

if __name__ == "__main__":
    generate("Iris-nikicus", 6.5, 7, 1.5, 3.5, "iris_modified_proof.data", 20)